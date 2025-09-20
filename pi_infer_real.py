# %%
import os
import time
from multiprocessing.managers import SharedMemoryManager

import pygame
import click
import cv2
import yaml
import numpy as np
import copy
import json
from scipy.spatial.transform import Rotation as R
from omegaconf import OmegaConf
import torch
from common.precise_sleep import precise_wait
from real.controller_robot_system import ControllerRobotSystem
from real.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from common.real_inference_util import get_real_hra_obs_dict, get_real_hra_action
from openpi_client import image_tools
from openpi_client import websocket_client_policy



def right_pygame_show(color_image, camera_img_window, text, obs_res, is_right=True):
    cv2.putText(color_image, text, (10,20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=1.0, lineType=cv2.LINE_AA, thickness=2, color=(0,1,0))
    color_pygame = pygame.surfarray.make_surface(color_image.swapaxes(0, 1))
    if not is_right:
        camera_img_window.blit(color_pygame, (0, 0)) 
    else:
        camera_img_window.blit(color_pygame, (obs_res[0], 0)) 
    pygame.display.update()


def human_loop_control(env, camera_img_window, key_counter, obs_res, dt, instruction):

    t_start = time.monotonic()
    time.sleep(0.4)
    iter_idx = 0
    while True:
        t_cycle_end = t_start + (iter_idx + 1) * dt
        obs = env.get_obs()
        color_image = obs['camera0_rgb'][-1]  # color, color_right, pointcloud (optional)

        # visualize current-obs
        right_pygame_show(color_image, camera_img_window, "human-loop: left-observation", obs_res, is_right=False)
        press_events = key_counter.get_press_events()
        start_policy = False
        press_events = press_events[:1]                  # only consider the first key stroke
        for key_stroke in press_events:
            if key_stroke == KeyCode(char='q'):          # exit the program
                env.end_episode(inference_mode=True)
                exit(0)
            elif key_stroke == KeyCode(char='c'):        # c: continue the policy inference
                start_policy = True
            elif key_stroke == KeyCode(char='r'):        # r: reset the robot (with match episode t = 0)
                env.robots[0].reset_pose()
                env.grippers[0].reset_pose()
                print("reset waiting finish.")
            elif key_stroke == KeyCode(char='i'):
                 # change task 
                instruction = input("print new instruction: ")
                instruction = instruction.strip()
                print("Switch to instruction: ", instruction)
            break
        if start_policy:
            break
        precise_wait(t_cycle_end)

    return instruction


def get_pi_obs_dict(obs_dict_np, instruction, robot_config, action_dim):

    obs_dict_pi = dict()
    n_robot = len(robot_config['robots'])
    n_gripper = len(robot_config['grippers'])
    assert n_robot == n_gripper
    # key_sequence = ['eef_pos', 'eef_rot_axis_angle', 'gripper_pose']
    # low_dim_dict = dict()
    # for key in key_sequence: 
    #     low_dim_dict[key] = list()
    states = []
    for i in range(n_robot):
        robot_prefix = "robot" + str(i)
        eef_pos = obs_dict_np[robot_prefix + '_eef_pos'][:-1]
        eef_rot_axis_angle = obs_dict_np[robot_prefix + '_eef_rot_axis_angle'][:-1]
        gripper_prefix = "gripper" + str(i)
        gripper_pose = obs_dict_np[gripper_prefix + '_gripper_pose'][1:]
        states.append(np.concatenate([eef_pos.flatten(), eef_rot_axis_angle.flatten(), gripper_pose.flatten()]))
    obs_dict_pi['state'] = np.concatenate(states).astype(np.float32)
    if action_dim is not None:
        if obs_dict_pi['state'].shape[0] < action_dim:
            pad_length = action_dim - obs_dict_pi['state'].shape[0]
            obs_dict_pi['state'] = np.concatenate([obs_dict_pi['state'], np.zeros(pad_length, dtype=np.float32)])
    obs_dict_pi['prompt'] = instruction
    images_seq = obs_dict_np['camera0_rgb']
    for i in range(len(images_seq)):
        img = images_seq[i].transpose(1, 2, 0)
        # obs_dict_pi['image_{}'.format(i + 1)] = image_tools.convert_to_uint8(img * 255.0)
        obs_dict_pi['image_{}'.format(i + 1)] = img * 2.0 - 1.0
        # obs_dict_pi['image_{}'.format(i + 1)] = img
        # obs_dict_pi['image_{}'.format(i + 1)] = img * 255.0
        # print(obs_dict_pi['image_{}'.format(i + 1)].max(), obs_dict_pi['image_{}'.format(i + 1)].min(), '<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    return obs_dict_pi


@click.command()
@click.option('--task_jsonl_fp', '-t', required=True, type=str, help='Directory to save task.jsonl instruction file')
@click.option('--output', '-o', required=True, help='Directory to save recording of evaluation.')
@click.option('--hand_to_eef_file', '-ehf', required=True)
@click.option('--pi_config', '-rc', required=True, help='Path to pi_config yaml file')
@click.option('--robot_config', '-rc', required=True, help='Path to robot_config yaml file')
@click.option('--init_joints', '-j', is_flag=True, default=True,
              help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--frequency', '-f', default=20, type=float, help="Control frequency in Hz.")
@click.option('--control_freq_downsample', '-cfd', default=1, type=int, help="Downsample real control / schedule frequency in Hz.")
@click.option('--resize_observation_resolution', '-ror', default="1280x720", type=str)
@click.option('--observation_resolution', '-or', default="640x480", help="Observation resolution.")
@click.option('--camera_exposure', '-ce', default=None, type=int)
@click.option('--robot_action_horizon', '-rah', default=12, type=int, help="Action horizon for model prediction.")
@click.option('--robot_steps_per_inference', '-rsi', default=6, type=int, help="Schedule waypoints per X timestamps.")
@click.option('--gripper_action_horizon', '-gah', default=12, type=int, help="Action horizon for model prediction.")
@click.option('--gripper_steps_per_inference', '-gsi', default=6, type=int, help="Schedule waypoints per X timestamps.")
@click.option('--ignore_start_chunk', '-isc', default=2, type=int, help="Number of ignored chunk when schedulet waypoints.")
@click.option('--temporal_agg', is_flag=True, default=False)
@click.option('--ensemble_steps', type=int, default=8)
@click.option('--ensemble_weights_exp_k', type=float, default=-0.1)
@click.option('--verbose', is_flag=True, default=False)
def main(task_jsonl_fp, output, hand_to_eef_file, pi_config, robot_config,
         init_joints,
         frequency,
         control_freq_downsample,
         resize_observation_resolution,
         observation_resolution,
         camera_exposure,
         robot_action_horizon,
         robot_steps_per_inference,
         gripper_action_horizon,
         gripper_steps_per_inference,
         ignore_start_chunk,
         temporal_agg,
         ensemble_steps,
         ensemble_weights_exp_k,
         verbose
         ):
    
    global TASK_ID
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    
    # load robot config file
    robot_config_data = yaml.safe_load(open(os.path.expanduser(robot_config), 'r'))

    robots_config = robot_config_data['robots']
    grippers_config = robot_config_data['grippers']
    camera_config = robot_config_data['cameras']
    active_vision_config = robot_config_data['active_visions']
    calib_cam2base = np.array(camera_config[0]["calib_cam_to_base"][0][1])
    print("calib_cam2base", calib_cam2base)
    
    n_total_action_dim = 0
    for i in range(len(robots_config)):
        robots_config[i]['action_dim_real'] = [n_total_action_dim, n_total_action_dim + robots_config[i]['action_dim']]
        n_total_action_dim = n_total_action_dim + robots_config[i]['action_dim']
        grippers_config[i]['action_dim_real'] = [n_total_action_dim, n_total_action_dim + grippers_config[i]['action_dim']]
        n_total_action_dim = n_total_action_dim + grippers_config[i]['action_dim']

    policy_client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)
    print("Establish Client Connection to PI0 Policy")

    ############################################################################################################################

    # setup experiment (for pi0)
    n_robots = len(robot_config_data['robots'])
    hand_to_eef = np.load(hand_to_eef_file)
    obs_res = tuple(int(x) for x in observation_resolution.split('x'))
    resize_res = tuple(int(x) for x in resize_observation_resolution.split('x'))
    with open(pi_config, "r") as f:
        pi_config = json.load(f)
    img_obs_shape = pi_config['img_obs_shape']
    camera_obs_horizon = pi_config['camera_obs_horizon']
    robot_obs_horizon = pi_config['robot_obs_horizon']
    gripper_obs_horizon = pi_config['gripper_obs_horizon']
    cfg_path = pi_config["config_file"]
    obs_downsample_steps = pi_config["obs_downsample_steps"]
    action_downsample_steps = pi_config["action_downsample_steps"]
    action_dim = pi_config["action_dim"]
    action_dim = None
    dt = 1 / frequency * action_downsample_steps
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # NOTE / TODO: Warning, current code only support 1 robot arm and 1 gripper for PI0 model.
    n_robot = len(robot_config_data['robots'])
    n_gripper = len(robot_config_data['grippers'])
    assert n_robot == 1
    assert n_gripper == 1

    ############################################################################################################################

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
                down_sample_steps=obs_downsample_steps,
                # obs
                camera_obs_horizon=camera_obs_horizon,
                robot_obs_horizon=robot_obs_horizon,
                gripper_obs_horizon=gripper_obs_horizon,
                active_vision_obs_horizon=0,
                # action
                init_joints=init_joints,
                # vis params, we vis the resut in the main loop
                enable_multi_cam_vis=False,
                verbose=verbose,
                shm_manager=shm_manager) as env:

            cv2.setNumThreads(2)
            print("Waiting for camera")
            time.sleep(1.0) 
            print('Ready!')

            # warmup
            # test the inference process of model
            obs_pose_rep = pi_config['obs_pose_rep']          
            action_pose_repr = pi_config['action_pose_rep']   
            print("Warming up policy inference")
            obs = env.get_obs()
            obs_dict_np = get_real_hra_obs_dict(
                env_obs=obs, shape_meta=cfg['shape_meta'],
                obs_pose_repr=obs_pose_rep,
                img_obs_shape=img_obs_shape,
                calib_cam2base=calib_cam2base,
                hand_to_eef=hand_to_eef
                )
            instruction = "do something !"
            obs_dict_pi = get_pi_obs_dict(obs_dict_np, instruction, robot_config_data, action_dim)
            print(obs_dict_pi['state'])
            raw_action = policy_client.infer(obs_dict_pi)["actions"][0]
            print(raw_action[:,:3])
            # assert raw_action.shape[-1] == 15 * len(robots_config)
            action = get_real_hra_action(raw_action, obs_dict_np, action_pose_repr, calib_cam2base=calib_cam2base, hand_to_eef=hand_to_eef, n_robots=n_robots)
            action_horizon = action.shape[0]
            print(f'action_horizon is {action_horizon}')
            print(f'action_dim is {action_dim}')
            print('Ready!')

            max_timesteps = 2000
            all_time_actions = np.zeros((max_timesteps, max_timesteps + robot_action_horizon, 6))
            time.sleep(2.0)

            while True:
                # ========= human control loop ==========
                
                print("Human in the loop control.")
                print(f"Current instruction: {instruction}")
                print("- c: continue the policy inference; \n- r: reset the robot;\n- q: exit.\n- e: end the policy inference;\n- i: change instruction;\n")
                instruction = human_loop_control(env, camera_img_window, key_counter, obs_res, dt, instruction)
                print(f"Execute instruction: {instruction}")

                try:
                    print("simulate policy start.")
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
                    first_robot_infer = True
                    first_gripper_infer = True

                    while True:
                        s = time.time()

                        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        print(s)
                        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

                        press_events = key_counter.get_press_events()
                        if len(press_events) > 0:
                            key_stroke = press_events[0]                  # only consider the first key stroke
                            if key_stroke == KeyCode(char='e'):          # exit the program
                                print("Policy Inference Stop!")
                                env.end_episode(inference_mode=True)
                                break

                        t_cycle_end = t_start + (iter_idx + control_freq_downsample) * dt
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        print(f'Obs latency {time.time() - obs_timestamps[-1]}')
                        
                        right_pygame_show(obs['camera0_rgb'][-1], camera_img_window, "zed observation", obs_res, is_right=False)
                        right_pygame_show(255 * (obs_dict_pi['image_1'] + 1.0) / 2.0, camera_img_window, "policy observation", obs_res, is_right=True)
                        print(obs_dict_pi['prompt'])
                        print(obs_dict_pi['state'])


                        # policy-loop before while
                        obs_dict_np = get_real_hra_obs_dict(
                            env_obs=obs, shape_meta=cfg['shape_meta'],
                            obs_pose_repr=obs_pose_rep,
                            img_obs_shape=img_obs_shape,
                            calib_cam2base=calib_cam2base,
                            hand_to_eef=hand_to_eef
                            )
                        obs_dict_pi = get_pi_obs_dict(obs_dict_np, instruction, robot_config_data, action_dim)
                        raw_action = policy_client.infer(obs_dict_pi)["actions"][0]
                        print(raw_action[:,:3])
                        # assert raw_action.shape[-1] == 15 * len(robots_config)
                        action = get_real_hra_action(raw_action, obs_dict_np, action_pose_repr, calib_cam2base=calib_cam2base, hand_to_eef=hand_to_eef, n_robots=n_robots)
                        
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
                            print(len(this_target_poses))
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

                        # wait for execution
                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += control_freq_downsample
                        inference_idx += control_freq_downsample

                except KeyboardInterrupt:
                    print("KeyboardInterrupt")
                    env.end_episode(inference_mode=True)
                    exit(0)

# %%
if __name__ == '__main__':
    main()

# python infer_real_franka.py --output output_eval --robot_config=real/config/franka_inspire_atv_cam_unimanual.yaml -f 20  -ror 1280x720 -or 640x480
# -md output_proc/output_1280x720_640x480.zarr

