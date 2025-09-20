# %%
import os
import time
from multiprocessing.managers import SharedMemoryManager

import pygame
import click
import cv2
import json
import yaml
import numpy as np
import copy
import scipy.spatial.transform as st
from scipy.spatial.transform import Rotation as R
from omegaconf import OmegaConf
import omegaconf
import dill
import hydra
import torch
from common.replay_buffer import ReplayBuffer
from common.precise_sleep import precise_wait
from real.controller_robot_system import ControllerRobotSystem
from real.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from diffusion_policy.dataset.hra_dataset import get_real_sampler_idx, get_instruction_from_filename_list
from common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from common.real_inference_util import get_real_hra_obs_dict, get_real_hra_action
from real.keystroke_counter import (KeystrokeCounter, Key, KeyCode)


text_feature_bookmark = dict() 
clip_model = None
tokenizer = None
hand_shrink_coef = 1.0

def get_text_feature(text, text_feature_cache_dir):
    print("Switch to instruction: {}".format(text))

    # we use cache and bookmark to speed up the text feature extraction
    global text_feature_bookmark, clip_model, tokenizer

    text = text.replace('_', ' ').strip()
    if not text.endswith('.'):
        text = text + '.'

    if text in text_feature_bookmark:
        return text_feature_bookmark[text]
    
    text_filename = text.replace(" ", "_") 
    text_fp = os.path.join(text_feature_cache_dir, text_filename + 'npy')
    if os.path.exists(text_fp):
        text_feature = np.load(text_fp)
        text_feature_bookmark[text] = text_feature
        return text_feature

    device = 'cuda'
    if clip_model is None:
        import clip
        clip_model, _ = clip.load("ViT-B/16", device=device, jit=False)
        tokenizer = clip.tokenize

    text_tokens = tokenizer([text]).to(device)  
    text_features = clip_model.encode_text(text_tokens).detach().cpu().numpy()  # (1, 512)
    text_features = text_features[0]

    os.makedirs(text_feature_cache_dir, exist_ok=True)
    if not os.path.exists(text_fp):
        np.save(text_fp, text_features)
        print(f'Saved text feature for "{text}" to {text_fp}')
    text_feature_bookmark[text] = text_features
    return text_features


def right_pygame_show(color_image, camera_img_window, text, obs_res, is_right=True):
    cv2.putText(color_image, text, (10,20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=1.0, lineType=cv2.LINE_AA, thickness=2, color=(0,1,0))
    color_pygame = pygame.surfarray.make_surface(color_image.swapaxes(0, 1))
    if not is_right:
        camera_img_window.blit(color_pygame, (0, 0)) 
    else:
        camera_img_window.blit(color_pygame, (obs_res[0], 0)) 
    pygame.display.update()



def human_loop_control(env, camera_img_window, key_counter, obs_res, dt, dataset_list, task_id, instruction, instruction_list, record_trajectory):

    t_start = time.monotonic()
    # time.sleep(2.0)
    iter_idx = 0
    while True:
        t_cycle_end = t_start + (iter_idx + 1) * dt
        obs = env.get_obs()
        color_image = obs['camera0_rgb'][-1]  # color, color_right, pointcloud (optional)

        # visualize current-obs
        right_pygame_show(color_image, camera_img_window, "human-loop: left-observation", obs_res, is_right=False)
        if "camera0_rgb_right" in obs:
            color_image_right = obs['camera0_rgb_right'][-1]
            right_pygame_show(color_image_right, camera_img_window, "human-loop: right-observation", obs_res, is_right=True)
        press_events = key_counter.get_press_events()
        start_policy = False
        press_events = press_events[:1]                  # only consider the first key stroke
        for key_stroke in press_events:
            if key_stroke == KeyCode(char='q'):          # exit the program
                env.end_episode(inference_mode=(not record_trajectory))
                exit(0)
            elif key_stroke == KeyCode(char='c'):        # c: continue the policy inference
                start_policy = True
            elif key_stroke == KeyCode(char='r'):        # r: reset the robot (with match episode t = 0)
                env.robots[0].reset_pose()
                env.grippers[0].reset_pose()
                time.sleep(0.5)
                print("reset waiting finish.")
            elif key_stroke == KeyCode(char='t'):
                if dataset_list is None:
                    # print("use_instruction=True, only instruction-mode is supported.")
                    pass
                else:
                    # change task 
                    print("Task-ID:")
                    for i in range(len(dataset_list)):
                        print(f"{i}: {dataset_list[i]}")
                    task_id = input("input task id: ")
                    n_str = len(task_id)
                    st = 0
                    for i in range(n_str):
                        if task_id[i] < '0' or task_id[i] > '9':
                            st = i + 1 
                        else:
                            break
                    task_id = int(task_id[st:])
                    instruction = instruction_list[task_id]
                    print("Switch to task id: ", task_id)
                    print("Switch to task name: ", instruction)
            elif key_stroke == KeyCode(char='i'):
                if dataset_list is not None:
                    # print("use_instruction=False, only task-id-mode is supported.")
                    pass
                else:
                    instruction = input("input instruction: ")
                    instruction = instruction.strip()
                    if not instruction.endswith("."):
                        instruction += "."
                    print("Switch to instruction: ", instruction)
        if start_policy:
            break
        precise_wait(t_cycle_end)
    return task_id, instruction


@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save recording of evaluation.')
@click.option('--text_feature_cache_dir', '-tcd', required=True, help='Directory to save recording of evaluation.')
@click.option('--hand_to_eef_file', '-ehf', required=True)
@click.option('--robot_config', '-rc', required=True, help='Path to robot_config yaml file')
@click.option('--dataset_list_fp', '-d', default=None, help='Path to dataset_list.json')
@click.option('--init_joints', '-j', is_flag=True, default=True,
              help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--frequency', '-f', default=20, type=float, help="Control frequency in Hz.")
@click.option('--control_freq_downsample', '-cfd', default=1, type=int, help="Downsample real control frequency in Hz.")
@click.option('--resize_observation_resolution', '-ror', default="1280x720", type=str)
@click.option('--observation_resolution', '-or', default="640x480", help="Observation resolution.")
@click.option('--camera_exposure', '-ce', default=None, type=int)
@click.option('--enable_pointcloud', '-ep', is_flag=True, default=False)
@click.option('--robot_action_horizon', '-rah', default=12, type=int, help="Action horizon for model prediction.")
@click.option('--robot_steps_per_inference', '-rsi', default=6, type=int, help="Schedule waypoints per X timestamps.")
@click.option('--gripper_action_horizon', '-gah', default=12, type=int, help="Action horizon for model prediction.")
@click.option('--gripper_steps_per_inference', '-gsi', default=6, type=int, help="Schedule waypoints per X timestamps.")
@click.option('--ignore_start_chunk', '-isc', default=2, type=int, help="Number of ignored chunk when schedulet waypoints.")
@click.option('--temporal_agg', is_flag=True, default=False)
@click.option('--ensemble_steps', type=int, default=8)
@click.option('--ensemble_weights_exp_k', type=float, default=-0.1)
@click.option('--use_predefine_instruction', is_flag=True, default=False)
@click.option('--record_trajectory', is_flag=True, default=False)
@click.option('--verbose', is_flag=True, default=False)
def main(input, output, text_feature_cache_dir, 
         hand_to_eef_file, robot_config,
         dataset_list_fp,
         init_joints,
         frequency,
         control_freq_downsample,
         resize_observation_resolution,
         observation_resolution,
         camera_exposure,
         enable_pointcloud,
         robot_action_horizon,
         robot_steps_per_inference,
         gripper_action_horizon,
         gripper_steps_per_inference,
         ignore_start_chunk,
         temporal_agg,
         ensemble_steps,
         ensemble_weights_exp_k,
         use_predefine_instruction,
         record_trajectory,
         verbose
         ):
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    assert robot_steps_per_inference % control_freq_downsample == 0
    assert gripper_steps_per_inference % control_freq_downsample == 0

    # load robot config file
    robot_config_data = yaml.safe_load(open(os.path.expanduser(robot_config), 'r'))

    robots_config = robot_config_data['robots']
    grippers_config = robot_config_data['grippers']
    active_vision_config = robot_config_data['active_visions']
    camera_config = robot_config_data['cameras']
    calib_cam2base = np.array(camera_config[0]["calib_cam_to_base"][0][1])
    
    print("calib_cam2base", calib_cam2base)

    n_total_action_dim = 0
    for i in range(len(robots_config)):
        robots_config[i]['action_dim_real'] = [n_total_action_dim, n_total_action_dim + robots_config[i]['action_dim']]
        n_total_action_dim = n_total_action_dim + robots_config[i]['action_dim']
        grippers_config[i]['action_dim_real'] = [n_total_action_dim, n_total_action_dim + grippers_config[i]['action_dim']]
        n_total_action_dim = n_total_action_dim + grippers_config[i]['action_dim']

    # load checkpoint
    ckpt_path = input
    if not ckpt_path.endswith('.ckpt'):
        ckpt_path = os.path.join(ckpt_path, 'checkpoints', 'latest.ckpt')
    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)

    cfg = payload['cfg']
    print("model_name:", cfg.policy.obs_encoder.model_name)
    print("dataset_path:", cfg.task.dataset.dataset_path)


    ############################################################################################################################

    # setup experiment
    n_robots = len(robot_config_data['robots'])
    hand_to_eef = np.load(hand_to_eef_file)
    obs_res = tuple(int(x) for x in observation_resolution.split('x'))
    resize_res = tuple(int(x) for x in resize_observation_resolution.split('x'))
    # assert cfg.task.obs_down_sample_steps == 1, "obs_down_sample_steps must be 1 for real inference"
    # assert frequency == 20
    frequency = frequency
    obs_downsample_steps = cfg.task.obs_down_sample_steps
    action_downsample_steps = cfg.task.action_down_sample_steps
    dt = 1 / frequency * action_downsample_steps
    print("frequency:", frequency)
    print("dt:", dt)
    print("obs_downsample_steps:", obs_downsample_steps)
    print("action_downsample_steps:", action_downsample_steps)
    print("obs_res:", obs_res)
    print("resize_res:", resize_res)
    img_obs_shape = cfg.shape_meta.obs.camera0_rgb.shape[1:]
    camera_obs_horizon = cfg.task.img_obs_horizon
    robot_obs_horizon = cfg.task.low_dim_obs_horizon
    gripper_obs_horizon = cfg.task.low_dim_obs_horizon
    active_vision_obs_horizon = cfg.task.low_dim_obs_horizon
    use_instruction = cfg.task.use_instruction if hasattr(cfg.task, 'use_instruction') else False
    last_task_instruction = "do something."
    ############################################################################################################################

    if use_instruction:
        if use_predefine_instruction:
            if dataset_list_fp is not None:
                with open(dataset_list_fp, 'r') as f:
                    dataset_list = json.load(f)
            dataset_list = [fp.split('/')[-1] for fp in dataset_list]
            instruction_list = get_instruction_from_filename_list(dataset_list)
        else:
            dataset_list = None
    else:
        if dataset_list_fp is not None:
            with open(dataset_list_fp, 'r') as f:
                dataset_list = json.load(f)
        dataset_list = [fp.split('/')[-1] for fp in dataset_list]
        instruction_list = get_instruction_from_filename_list(dataset_list)
    task_id = 0
    instruction = "do something."



    pygame.init()
    camera_img_window = pygame.display.set_mode((obs_res[0]*2, obs_res[1]))  
    pygame.display.set_caption("evaluation_display")


    with (SharedMemoryManager() as shm_manager, \
          KeystrokeCounter() as key_counter):
        with ControllerRobotSystem(
                output_dir=output,
                cameras_config=camera_config,
                robots_config=robots_config,
                grippers_config=grippers_config,
                active_visions_config=active_vision_config,
                # obs
                frequency=frequency,
                obs_image_resize_resolution=resize_res,
                obs_image_resolution=obs_res,
                camera_exposure=camera_exposure,
                enable_pointcloud=enable_pointcloud,
                down_sample_steps=obs_downsample_steps,
                # obs
                camera_obs_horizon=camera_obs_horizon,
                robot_obs_horizon=robot_obs_horizon,
                gripper_obs_horizon=gripper_obs_horizon,
                active_vision_obs_horizon=active_vision_obs_horizon,
                # action
                init_joints=init_joints,
                # vis params, we vis the resut in the main loop
                enable_multi_cam_vis=False,
                shm_manager=shm_manager,
                verbose=verbose) as env:

            cv2.setNumThreads(2)
            print("Waiting for camera")
            time.sleep(1.0) 
            print('Ready!')

            # warmup
            # creating model
            # have to be done after fork to prevent
            # duplicating CUDA context with ffmpeg nvenc
            cls = hydra.utils.get_class(cfg._target_)
            workspace = cls(cfg)
            workspace: BaseWorkspace
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)

            policy = workspace.model
            if cfg.training.use_ema:
                policy = workspace.ema_model
            policy.num_inference_steps = 16  # DDIM inference iterations
            obs_pose_rep = cfg.task.pose_repr.obs_pose_repr              # obs_pose_rep: relative
            action_pose_repr = cfg.task.pose_repr.action_pose_repr       # action_pose_repr: relative
            
            print('obs_pose_rep', obs_pose_rep)
            print('action_pose_repr', action_pose_repr)

            device = torch.device('cuda')
            policy.eval().to(device)


            print("Warming up policy inference")
            obs = env.get_obs()
            with torch.no_grad():
                policy.reset()
                if use_instruction:
                    instruction = "do something."
                    text_feature = get_text_feature(instruction, text_feature_cache_dir)
                    obs['instruction'] = text_feature
                else:
                    obs['task_id'] = np.zeros((1, 1))
                obs_dict_np = get_real_hra_obs_dict(
                    env_obs=obs, shape_meta=cfg.task.shape_meta,
                    obs_pose_repr=obs_pose_rep,
                    img_obs_shape=img_obs_shape,
                    calib_cam2base=calib_cam2base,
                    hand_to_eef=hand_to_eef)
                obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device),
                                      not_use_keys=['robot0_ego_pose_mat', 'robot1_ego_pose_mat'])

                result = policy.predict_action(obs_dict)
                raw_action = result['action_pred'][0].detach().to('cpu').numpy()
                assert raw_action.shape[-1] == 15 * len(robots_config)
                action = get_real_hra_action(raw_action, obs_dict_np, action_pose_repr, calib_cam2base=calib_cam2base, hand_to_eef=hand_to_eef, n_robots=n_robots)
                action_dim = action.shape[-1]
                print(f'action_dim is {action_dim}')
                print(raw_action[:, :3])
                assert action.shape[-1] == 12 * len(robots_config)
                del result
            print('Ready!')

            max_timesteps = 2000
            all_time_actions = np.zeros((max_timesteps, max_timesteps + robot_action_horizon, 6))

            time.sleep(2.0)

            while True:
                # ========= human control loop ==========
                
                print("Human in the loop control.")
                if not use_instruction:
                    print(f"Current task: [{task_id}] {dataset_list[task_id]}")
                else:
                    print(f"Current instruction: {instruction}")
                if use_instruction and (not use_predefine_instruction):
                    print("- c: continue the policy inference; \n- r: reset the robot;\n- q: exit.\n- e: end the policy inference;\n- i: change instruction;")
                else:
                    print("- c: continue the policy inference; \n- r: reset the robot;\n- q: exit.\n- e: end the policy inference;\n- t: change task_id;")
                task_id, instruction = human_loop_control(env, camera_img_window, key_counter, obs_res, dt, dataset_list, task_id, instruction, instruction_list, record_trajectory)
                if use_instruction and instruction != last_task_instruction:
                    last_task_instruction = instruction
                    text_feature = get_text_feature(instruction, text_feature_cache_dir)

                try:
                    print("simulate policy start.")

                    # start episode
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)

                    obs = env.get_obs()

                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency
                    frame_latency = 1 / 60
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0
                    inference_idx = 0

                    # policy-loop before while
                    first_robot_infer = True
                    first_gripper_infer = True
                    while True:

                        press_events = key_counter.get_press_events()
                        if len(press_events) > 0:
                            key_stroke = press_events[0]                  # only consider the first key stroke
                            if key_stroke == KeyCode(char='e'):          # exit the program
                                print("Policy Inference Stop!")
                                env.end_episode(inference_mode=(not record_trajectory))
                                break

                        # calculate timing, use action_downsample_steps to downsample the action frequency
                        t_cycle_end = t_start + (iter_idx + control_freq_downsample) * dt

                        # get obs
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        print(f'Obs latency {time.time() - obs_timestamps[-1]}')
                        if use_instruction:
                            print(f"Instruction: {instruction}")
                        else:
                            print(f"Task ID: {task_id}")


                        color_image = obs['camera0_rgb'][-1]  # color, color_right, pointcloud (optional)
                        right_pygame_show(color_image, camera_img_window, "human-loop: left-observation", obs_res, is_right=False)
                        if 'camera0_rgb_right' in obs:
                            color_image_right = obs['camera0_rgb_right'][-1]
                            right_pygame_show(color_image_right, camera_img_window, "human-loop: right-observation", obs_res, is_right=True)

                        # run inference
                        with torch.no_grad():
                            s = time.time()
                            if use_instruction:
                                obs['instruction'] = text_feature
                            else:
                                obs['task_id'] = np.ones((1, 1)) * task_id
                            
                            obs_dict_np = get_real_hra_obs_dict(
                                env_obs=obs, shape_meta=cfg.task.shape_meta,
                                obs_pose_repr=obs_pose_rep,
                                img_obs_shape=img_obs_shape,
                                calib_cam2base=calib_cam2base,
                                hand_to_eef=hand_to_eef)
                            obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device),
                                                not_use_keys=['robot0_ego_pose_mat', 'robot1_ego_pose_mat'])
                            result = policy.predict_action(obs_dict)
                            action_raw = result['action_pred'][0].detach().to('cpu').numpy()

                            # action_raw = action_template.copy()

                            assert action_raw.shape[-1] == 15 * len(robots_config)
                            action = get_real_hra_action(action_raw, obs_dict_np, action_pose_repr, calib_cam2base=calib_cam2base, hand_to_eef=hand_to_eef, n_robots=n_robots)
                            print('Inference latency:', time.time() - s)

                        curr_time = time.time()
                        robot_action = []
                        for device_cfg in robots_config:
                            robot_action.append(action[:robot_action_horizon, device_cfg['action_dim_real'][0]: device_cfg['action_dim_real'][1]])
                        robot_action = np.concatenate(robot_action, axis=-1)
                        all_time_actions[[iter_idx], iter_idx:iter_idx + robot_action_horizon] = robot_action

                        if inference_idx % robot_steps_per_inference == 0:

                            if temporal_agg:
                                # temporal ensemble
                                action_seq_for_curr_step = all_time_actions[:, iter_idx:iter_idx + robot_action_horizon]
                                target_pose_list = []
                                for i in range(robot_action_horizon):
                                    st_idx = max(0, iter_idx - ensemble_steps + control_freq_downsample)
                                    ed_idx = iter_idx
                                    n_use_step = (ed_idx - st_idx) // control_freq_downsample
                                    use_idx = [ed_idx - i * control_freq_downsample for i in range(n_use_step + 1)]
                                    actions_for_curr_step = action_seq_for_curr_step[use_idx, i]
                                    actions_populated = np.all(actions_for_curr_step != 0, axis=1)
                                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                                    k = ensemble_weights_exp_k
                                    exp_weights = np.exp(k * np.arange(len(actions_for_curr_step)))
                                    exp_weights = exp_weights / exp_weights.sum()
                                    weighted_rotvec = R.from_rotvec(np.array(actions_for_curr_step)[:, 3:6]).mean(
                                        weights=exp_weights).as_rotvec()
                                    weighted_action = (actions_for_curr_step * exp_weights[:, np.newaxis]).sum(axis=0,
                                                                                                            keepdims=True)
                                    weighted_action[0][3:6] = weighted_rotvec
                                    target_pose_list.append(weighted_action)
                                this_target_poses = np.concatenate(target_pose_list, axis=0)
                            else:
                                this_target_poses = robot_action

                            action_timestamps = (np.arange(len(action), dtype=np.float64)) * dt + obs_timestamps[-1]
                            action_timestamps = action_timestamps[:robot_action_horizon]
                            print(f"Robot inference time: {inference_idx}, action_shape: {this_target_poses.shape}" )

                            is_new = action_timestamps > curr_time
                            if np.sum(is_new) == 0:
                                raise ValueError(f"Robot action over budget, curr_time: {curr_time}, action_timestamps: {action_timestamps}")
                            else:
                                this_target_poses = this_target_poses[is_new]
                                action_timestamps = action_timestamps[is_new]

                            if first_robot_infer is True:
                                action_timestamps = action_timestamps + 0.5
                                first_robot_infer = False
                                
                            # execute robot arm actions (we recommand gripper_steps_per_inference < gripper_action_horizon for franka robot)
                            # print(len(this_target_poses))
                            # import pdb; pdb.set_trace()
                            env.exec_robot(
                                robot_actions=this_target_poses,
                                timestamps=action_timestamps,
                                compensate_latency=True,
                                ignore_t=ignore_start_chunk,
                            )

                        if inference_idx % gripper_steps_per_inference == 0:
                            this_target_poses = []
                            for device_cfg in grippers_config:
                                this_target_poses.append(action[:gripper_action_horizon, device_cfg['action_dim_real'][0]: device_cfg['action_dim_real'][1]])
                            this_target_poses = np.concatenate(this_target_poses, axis=-1)

                            delta_target_poses = this_target_poses - this_target_poses[0]
                            this_target_poses = hand_shrink_coef * delta_target_poses + this_target_poses[0]

                            action_timestamps = (np.arange(len(action), dtype=np.float64)) * dt + obs_timestamps[-1]
                            action_timestamps = action_timestamps[:gripper_action_horizon]
                            print(f"Gripper inference time: {inference_idx}, action_shape: {this_target_poses.shape}" )

                            is_new = action_timestamps > curr_time
                            if np.sum(is_new) == 0:
                                raise ValueError(f"Gripper action over budget, curr_time: {curr_time}, action_timestamps: {action_timestamps}")
                            else:
                                this_target_poses = this_target_poses[is_new]
                                action_timestamps = action_timestamps[is_new]

                            if first_gripper_infer is True:
                                action_timestamps = action_timestamps + 0.5
                                first_gripper_infer = False
                                
                            # execute gripper actions (we recommand gripper_steps_per_inference == gripper_action_horizon for inspire hand)
                            env.exec_gripper(
                                robot_actions=this_target_poses,
                                timestamps=action_timestamps,
                                compensate_latency=True,
                                ignore_t=ignore_start_chunk,
                            )

                        print(time.time())

                        # wait for execution
                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += control_freq_downsample
                        inference_idx += control_freq_downsample


                except KeyboardInterrupt:
                    print("Policy Inference Stop!")
                    env.end_episode(inference_mode=(not record_trajectory))
                    continue

# %%
if __name__ == '__main__':
    main()

# python infer_real_franka.py --output output_eval --robot_config=real/config/franka_inspire_atv_cam_unimanual.yaml -f 20  -ror 1280x720 -or 640x480
# -md output_proc/output_1280x720_640x480.zarr

