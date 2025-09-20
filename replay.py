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
import scipy.spatial.transform as st
import scipy.spatial.transform.rotation as R
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
from common.pose_util import pose_to_mat, mat_to_pose
from common.real_inference_util import get_real_hra_obs_dict, get_replay_hra_action
from common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from real.keystroke_counter import (KeystrokeCounter, Key, KeyCode)



global_match_episode_id = None
global_match_replay_buffer = None
match_episode = None
def get_match_episode(match_data_dir):
    global global_match_replay_buffer, match_episode
    match_episode = None 
    print("Loading match episode, please wait...")
    if (match_data_dir is not None) and match_data_dir.endswith(".zarr"):
        if global_match_replay_buffer is None:
            global_match_replay_buffer = ReplayBuffer.create_from_path(zarr_path=match_data_dir, mode='r')
        if match_episode is None:
            match_episode = global_match_replay_buffer.get_episode(int(global_match_episode_id))
            print(f"Get match episode {global_match_episode_id} from replay buffer, length of episode: {len(match_episode['robot0_eef_pos'])}")
        return len(match_episode['timestamp'])
    else:
        raise ValueError("match_data_dir is illegal, should be .zarr folder. currently: {}".format(match_data_dir))


def right_pygame_show(color_image, camera_img_window, text, obs_res, is_right=True):
    cv2.putText(color_image, text, (10,20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=1.0, lineType=cv2.LINE_AA, thickness=2, color=(255, 0, 0))
    color_pygame = pygame.surfarray.make_surface(color_image.swapaxes(0, 1))
    if not is_right:
        camera_img_window.blit(color_pygame, (0, 0)) 
    else:
        camera_img_window.blit(color_pygame, (obs_res[0], 0)) 
    pygame.display.update()



def human_loop_control(robot_config_data, cfg, hand_to_eef, calib_cam2base, env, camera_img_window, key_counter, match_data_dir, resize_res, obs_res, dt, command_latency):

    global match_episode, global_match_episode_id

    if match_data_dir is not None:
        episode_length = get_match_episode(match_data_dir)

    t_start = time.monotonic()
    iter_idx = 0
    replay_episode_iter = -1
    start_replay = False

    while True:

        t_cycle_end = t_start + (iter_idx + 1) * dt
        t_sample = t_cycle_end - command_latency
        t_command_target = t_cycle_end + dt
        obs = env.get_obs()
        color_image = obs['camera0_rgb'][-1]  # color, color_right, pointcloud (optional)

        # visualize current-obs
        right_pygame_show(color_image, camera_img_window, "human-loop: left-observation", obs_res, is_right=False)

        press_events = key_counter.get_press_events()
        start_policy = False

        if match_episode is None:
            right_pygame_show(255 * np.ones_like(color_image).astype(np.uint8), camera_img_window, f"no match episode: dir({match_data_dir})", obs_res)
        elif replay_episode_iter == -1:
            right_pygame_show(match_episode['camera0_rgb'][0], camera_img_window, f"match episode {global_match_episode_id}: ts=0", obs_res)
        elif replay_episode_iter + 1 == episode_length:
            right_pygame_show(match_episode['camera0_rgb'][replay_episode_iter], camera_img_window, f"match episode {global_match_episode_id}: replay finished", obs_res)
        else:
            right_pygame_show(match_episode['camera0_rgb'][replay_episode_iter], camera_img_window, f"match episode {global_match_episode_id}: ts={replay_episode_iter}", obs_res)

        press_events = press_events[:1]                  # only consider the first key stroke
        for key_stroke in press_events:
            if key_stroke == KeyCode(char='q'):          # exit the program
                replay_episode_iter = -1
                start_replay = False
                env.end_episode(inference_mode=True)
                exit(0)
            elif key_stroke == KeyCode(char='e'):        # e: match next episode
                replay_episode_iter = -1
                if (match_data_dir is not None) and (global_match_episode_id is not None) and (global_match_replay_buffer is not None):
                    global_match_episode_id = min(int(global_match_episode_id) + 1, global_match_replay_buffer.n_episodes - 1)
                    match_episode = None
                    episode_length = get_match_episode(match_data_dir)
                start_replay = False
            elif key_stroke == KeyCode(char='w'):        # w: match previous episode
                replay_episode_iter = -1
                if (match_data_dir is not None) and (global_match_episode_id is not None):
                    global_match_episode_id = max(int(global_match_episode_id) - 1, 0)
                    match_episode = None
                    episode_length = get_match_episode(match_data_dir)
                start_replay = False
            elif key_stroke == KeyCode(char='r'):        # r: reset the robot (with match episode t = 0)
                if match_episode is None:
                    raise NotImplementedError("Please implement a safe random reset mechanism here.")
                else:
                    duration = 4.0
                    length = len(match_episode[f'action'])
                    raw_action = match_episode[f'action'][0:1]
                    # raw_action[:, 6:12] = raw_action[:, 18:24]
                    obs_dict_np = get_real_hra_obs_dict(
                            env_obs=obs, shape_meta=cfg['shape_meta'],
                            obs_pose_repr='relative',
                            calib_cam2base=calib_cam2base,
                            hand_to_eef=hand_to_eef)
                    action = get_replay_hra_action(raw_action, obs_dict_np, 'abs', calib_cam2base=calib_cam2base, hand_to_eef=hand_to_eef, n_robots=1)
                    assert len(env.robots) == 1
                    assert len(env.grippers) == 1
                    for robot_idx in range(len(env.robots)):
                        env.robots[robot_idx].servoL(action[0, 12*robot_idx:12*robot_idx+6], duration=duration)
                        env.grippers[robot_idx].servoL(action[0, 12*robot_idx+6:12*robot_idx+12], duration=duration)
                    replay_episode_iter = 0
                    time.sleep(duration)
                    print("finished replaying initial pose")
                    t_start = time.monotonic()
                    iter_idx = 0
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt
                start_replay = False
            elif key_stroke == KeyCode(char='p'):        # r: replay the match episode step by step
                if replay_episode_iter == -1:
                    right_pygame_show(match_episode['camera0_rgb'][replay_episode_iter], camera_img_window, f"match episode {global_match_episode_id}: please reset (r) before replay (p).", obs_res)
                else:
                    start_replay = True
            elif key_stroke == KeyCode(char='e'):
                start_replay = False
        if start_policy:
            break
        
        if start_replay and (replay_episode_iter + 1 < episode_length) and (replay_episode_iter != -1):

            ##### Step-by-Step Replay
            actions = {"robot": [], "gripper": [], "active_vision": []}
            assert len(env.robots) == 1
            assert len(env.grippers) == 1
            raw_action = match_episode[f'action'][replay_episode_iter+1: replay_episode_iter+2]
            # raw_action[:, 6:12] = raw_action[:, 18:24]
            # raw_action[:, :6] = match_episode[f'action'][0:1, :6]
            obs_dict_np = get_real_hra_obs_dict(
                    env_obs=obs, shape_meta=cfg['shape_meta'],
                    obs_pose_repr='relative',
                    calib_cam2base=calib_cam2base,
                    hand_to_eef=hand_to_eef)
            action = get_replay_hra_action(raw_action, obs_dict_np, 'abs', calib_cam2base=calib_cam2base, hand_to_eef=hand_to_eef, n_robots=1)
            for robot_idx in range(len(env.robots)): 
                pose = action[:, 12*robot_idx:12*robot_idx+6]
                actions["robot"].append(pose)
                gripper_pose = action[:, 12*robot_idx+6:12*robot_idx+12]
                actions["gripper"] = [gripper_pose]
            
            # print(raw_action[:, :3] - np.array([0.15, 0.15, 0.15]))
            # print(action)

            precise_wait(t_sample)
            env.exec_actions(
                robot_actions=actions,
                timestamps=np.array([t_command_target - time.monotonic() + time.time()]),
                compensate_latency=False)
            replay_episode_iter = replay_episode_iter + 1

        precise_wait(t_cycle_end)
        iter_idx = iter_idx + 1



@click.command()
@click.option('--output', '-o', required=True, help='Directory to save recording of evaluation.')
@click.option('--hand_to_eef_file', '-ehf', required=True)
@click.option('--robot_config', '-rc', required=True, help='Path to robot_config yaml file')
@click.option('--init_joints', '-j', is_flag=True, default=True,
              help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--frequency', '-f', default=15, type=float, help="Control frequency in Hz.")
@click.option('--match_data_dir', '-md', default=None, type=str, help="Match-data from teleop for system validation.")
@click.option('--match_episode_id', '-me', default=0, type=str, help="Match-data from teleop for system validation.")
@click.option('--resize_observation_resolution', '-ror', default="1280x720", type=str)
@click.option('--observation_resolution', '-or', default="640x480", help="Observation resolution.")
@click.option('--command_latency', '-cl', default=0.01, type=float,
              help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
@click.option('--verbose', is_flag=True, default=False)
def main(output, hand_to_eef_file, robot_config,
         init_joints,
         frequency,
         match_data_dir, 
         match_episode_id,
         resize_observation_resolution,
         observation_resolution,
         command_latency,
         verbose
         ):
    
    embodiment = 0
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    
    global global_match_episode_id
    global_match_episode_id = match_episode_id

    # load robot config file
    robot_config_data = yaml.safe_load(open(os.path.expanduser(robot_config), 'r'))

    robots_config = robot_config_data['robots']
    grippers_config = robot_config_data['grippers']
    camera_config = robot_config_data['cameras']
    active_vision_config = robot_config_data['active_visions']
    calib_cam2base = np.array(camera_config[0]["calib_cam_to_base"][0][1])
    print("calib_cam2base", calib_cam2base)

    cfg_path = "diffusion_policy/config/task/dp.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)


    ############################################################################################################################

    # setup experiment
    n_robots = len(robot_config_data['robots'])
    hand_to_eef = np.load(hand_to_eef_file)
    obs_res = tuple(int(x) for x in observation_resolution.split('x'))
    resize_res = tuple(int(x) for x in resize_observation_resolution.split('x'))
    # assert cfg.task.obs_down_sample_steps == 1, "obs_down_sample_steps must be 1 for real inference"
    # assert frequency == 20
    frequency = frequency
    dt = 1 / frequency
    img_obs_shape = cfg['shape_meta']['obs']['camera0_rgb']['shape'][1:]
    camera_obs_horizon = cfg['img_obs_horizon']
    robot_obs_horizon = cfg['low_dim_obs_horizon']
    gripper_obs_horizon = cfg['low_dim_obs_horizon']
    active_vision_obs_horizon = cfg['low_dim_obs_horizon']
    ############################################################################################################################

    pygame.init()
    camera_img_window = pygame.display.set_mode((obs_res[0]*2, obs_res[1]))  
    pygame.display.set_caption("evaluation_display")


    with (SharedMemoryManager() as shm_manager, KeystrokeCounter() as key_counter):
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
                down_sample_steps=1,
                # obs
                camera_obs_horizon=camera_obs_horizon,
                robot_obs_horizon=robot_obs_horizon,
                gripper_obs_horizon=gripper_obs_horizon,
                active_vision_obs_horizon=active_vision_obs_horizon,
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

            while True:
                # ========= human control loop ==========
                
                try:
                    print("Human in the loop control.")
                    print("- e: match next episode;\n- w: match previous episode;\n- r: reset the robot to episode begin;\n- p: replay the match episode step by step.\n- q: exit.")
                    human_loop_control(robot_config_data, cfg, hand_to_eef, calib_cam2base, env, camera_img_window, key_counter, match_data_dir, resize_res, obs_res, dt, command_latency)
                except KeyboardInterrupt:
                    print("KeyboardInterrupt")
                    break

# %%
if __name__ == '__main__':
    main()
