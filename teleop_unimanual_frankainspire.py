# left-middle: start
# left-index: stop
# right-middle: save
# right-ring: delete

# # %%
import os
import pathlib
import time
from multiprocessing.managers import SharedMemoryManager

import av
import click
import cv2
# import open3d as o3d               # if import, will cause ZedCamera Initialization fail. Could not found GPU
import yaml
import dill
import hydra
import numpy as np
import scipy.spatial.transform as st
import scipy.spatial.transform.rotation as R
import torch
from omegaconf import OmegaConf
from common.precise_sleep import precise_wait
from real.controller_robot_system import ControllerRobotSystem
from real.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from common.pose_util import pose_to_mat, mat_to_pose
from pytransform3d import rotations
from real.keystroke_counter import (KeystrokeCounter, Key, KeyCode)
import pygame
import pygame.surfarray
from real.vuer_teleop import VuerTeleop

OmegaConf.register_new_resolver("eval", eval, replace=True)
record_condition = ["Not Recording (Left Middle Start)", "Is Recording (Right Middle End)", "Confirm Recording (Right Index Save, Left Index Delete)"]
finger_tip_index_dict = {"thumb": 4, "index": 9, "middle": 14, "ring": 19, "pinky": 24}
base_cam_hand_calib_rot = np.eye(3)
base_cam_hand_calib_trans = np.array([0., 0., 0.])
robot_default_init_pose = None
action_smooth_dict = None
finger_limit_robot_pose_mat = np.array([
    [1., 0., 0., 0.], 
    [0., 1., 0., 0.2], 
    [0., 0., 1., 0.], 
    [0., 0., 0., 1.]
])

tx_tip_flange =  np.array([[ 0.22442406, -0.97398856, -0.03130759,  0.34816485],
                            [-0.97148255, -0.22109284, -0.08567143,  0.35104975],
                            [ 0.0765211,   0.04964151, -0.99583146,  0.6070878 ],
                            [ 0.,          0.,          0.,          1.        ]])
tx_flange_tip = np.linalg.inv(tx_tip_flange)

# 将图像转换为 Pygame 可显示的格式
def cv2_to_pygame(cv_image):
    # OpenCV 是 BGR 格式，Pygame 是 RGB 格式，所以需要转换
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    return pygame.surfarray.make_surface(cv_image.swapaxes(0, 1))  # 使用swapaxes交换轴


def finger_tip_less_than_threshold(hand_mat, first_finger:str, second_finger:str, threshold):
    global finger_tip_index_dict
    first_finger_position = hand_mat[finger_tip_index_dict[first_finger]]
    second_finger_position = hand_mat[finger_tip_index_dict[second_finger]]
    dis = np.linalg.norm(first_finger_position - second_finger_position)
    if dis < threshold:
        return True
    else:
        return False
    # return dis < threshold

def check_hand_pose(hand_mat, first_finger:str, second_finger:str):
    if np.linalg.norm(hand_mat[1] - hand_mat[2]) == 0:
        positive_threshold = 0.1
        negative_threshold = 0.2
    else:
        positive_threshold = np.linalg.norm(hand_mat[1] - hand_mat[2]) / 1.5
        negative_threshold = np.linalg.norm(hand_mat[1] - hand_mat[2]) / 0.8
    finger_close_flag = finger_tip_less_than_threshold(hand_mat, first_finger, second_finger, positive_threshold)
    if finger_close_flag is False:
        return False

    global finger_tip_index_dict
    for finger_name, index in finger_tip_index_dict.items():
        if (finger_name is first_finger) or (finger_name is second_finger):
            continue
        else:
            finger_separate_flag = finger_tip_less_than_threshold(hand_mat, first_finger, finger_name, negative_threshold)

            if finger_separate_flag is True:
                return False
    return True

def draw_image_caption(color_img, color_img_right, text, position, 
                       fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, lineType=cv2.LINE_AA, thickness=2, color=(0,0,0)):
    cv2.putText(color_img, text, position,
                fontFace=fontFace,
                fontScale=fontScale,
                lineType=lineType,
                thickness=thickness,
                color=color)
    if color_img_right is not None:
        cv2.putText(color_img_right, text, position,
                    fontFace=fontFace,
                    fontScale=fontScale,
                    lineType=lineType,
                    thickness=thickness,
                    color=color)



def teleop_state_control(obs, 
                         env,
                         recording_flag,
                         episode_idx, ori_left_hand_mat, ori_right_hand_mat, ori_left_wrist_mat, ori_right_wrist_mat, human_robot_offset,
                         camera_img_window,
                         resolution,
                         verbose):

    global base_eye_hand_pos, robot_default_init_pose, tx_tip_flange, tx_flange_tip, base_cam_hand_calib_rot, base_cam_hand_calib_trans, action_smooth_dict

    color_img = obs['camera0_rgb'][-1]
    color_img_right = obs['camera0_rgb_right'][-1] if 'camera0_rgb_right' in obs else None
    draw_image_caption(color_img, color_img_right, f"Episode Index: episode {episode_idx}", (50, 30), color=(255, 0, 0))

    if recording_flag is record_condition[0]:
        color_caption = (255, 0, 255)
    elif recording_flag is record_condition[1]:
        color_caption = (0, 255, 255)
    else:
        color_caption = (255, 255, 0)
    draw_image_caption(color_img, color_img_right, f"{recording_flag}", (10, 20), color=color_caption)

    left_middle_thumb_pinch = check_hand_pose(ori_left_hand_mat, "thumb", "middle")
    left_ring_thumb_pinch = check_hand_pose(ori_left_hand_mat, "thumb", "ring")
    left_index_thumb_pinch = check_hand_pose(ori_left_hand_mat, "thumb", "index")
    right_middle_thumb_pinch = check_hand_pose(ori_right_hand_mat, "thumb", "middle")
    right_ring_thumb_pinch = check_hand_pose(ori_right_hand_mat, "thumb", "ring")
    right_index_thumb_pinch = check_hand_pose(ori_right_hand_mat, "thumb", "index")

    if verbose is True:
        draw_image_caption(color_img, color_img_right, f"left_middle_thumb_pinch is {left_middle_thumb_pinch}", (50, 120),
                           color=(255, 0, 0))
        draw_image_caption(color_img, color_img_right, f"left_ring_thumb_pinch is {left_ring_thumb_pinch}", (50, 160),
                           color=(255, 0, 0))
        draw_image_caption(color_img, color_img_right, f"left_index_thumb_pinch is {left_index_thumb_pinch}", (50, 200),
                           color=(255, 0, 0))
        draw_image_caption(color_img, color_img_right, f"right_middle_thumb_pinch is {right_middle_thumb_pinch}",
                           (50, 240), color=(255, 0, 0))
        draw_image_caption(color_img, color_img_right, f"right_ring_thumb_pinch is {right_ring_thumb_pinch}",
                           (50, 280), color=(255, 0, 0))
        draw_image_caption(color_img, color_img_right, f"right_index_thumb_pinch is {right_index_thumb_pinch}",
                           (50, 320), color=(255, 0, 0))

    start_record = False
    end_record = False
    if recording_flag != record_condition[2]:   # not wait for ensuring
        if left_middle_thumb_pinch is True:
            if verbose:
                print("start"*10)
            draw_image_caption(color_img, color_img_right, f"Begin", (50, 90), color=(255, 255, 0))
            start_record = True
        if left_index_thumb_pinch:
            if verbose:
                print("end" * 10)
            draw_image_caption(color_img, color_img_right, f"Confirm", (50, 90), color=(255, 255, 0))
            end_record = True
        action_smooth_dict = None

        if recording_flag is record_condition[1]:    # wait for ending
            if end_record is True:
                env.end_episode()
                recording_flag = record_condition[2] # start confirm
                # base_cam_hand_calib_rot = None
        if recording_flag is record_condition[0]:    # wait for starting
            if start_record is True:
                if robot_default_init_pose is None:
                    obs = env.get_obs()
                    if len(env.robots) > 0:
                        robot_default_init_pose = np.concatenate([obs['robot0_eef_pos'], obs['robot0_eef_rot_axis_angle']], axis=-1)[-1]
                
                # env.set_init_pose(robot_default_init_pose, duration=3.0, only_robot=True)
                obs = env.get_obs()
                # print("!!" * 1000)
                # base_cam_hand_calib = obs['robot0_eef_pos'][-1] - ori_right_wrist_mat[:3, -1]
                if len(env.robots) > 0:
                    init_franka_mat = pose_to_mat(np.concatenate([obs['robot0_eef_pos'], obs['robot0_eef_rot_axis_angle']], axis=-1)[-1])
                    base_cam_hand_calib_rot = np.linalg.inv(ori_right_wrist_mat[:3, :3]) @ init_franka_mat[:3, :3]
                    base_cam_hand_calib_trans = init_franka_mat[:3, -1] - ori_right_wrist_mat[:3, -1]
                env.start_episode()
                recording_flag = record_condition[1]
    else:
        if right_middle_thumb_pinch is True:
            if verbose:
                print("save" * 10)
            draw_image_caption(color_img, color_img_right, f"End (Save Last Time)", (50, 90), color=(255, 255, 0))
            end_record = True
        elif right_ring_thumb_pinch is True:
            if verbose:
                print("delete"*10)
            draw_image_caption(color_img, color_img_right, f"End (Delete Last Time)", (50, 80), color=(255, 255, 0))
            start_record = True
                        
        if end_record is True:
            recording_flag = record_condition[0]
        elif start_record is True:
            env.drop_episode() 
            recording_flag = record_condition[0]
        else:
            pass

    color_pygame = pygame.surfarray.make_surface(color_img.swapaxes(0, 1))

    camera_img_window.blit(color_pygame, (0, 0)) 
    pygame.display.update()

    for k in obs.keys():
        if k.startswith("camera") and k != "camera0_rgb":
            if obs[k][-1:].ndim == 2:
                depth_img_rgb = obs[k][-1:]
                depth_img_rgb = np.repeat(depth_img_rgb, 3, axis=0)  # copy 3 times, (3, H, W)
                depth_img = depth_img_rgb.transpose(2, 1, 0)
                depth_pygame = pygame.surfarray.make_surface(depth_img)
            else:
                depth_img = obs[k][-1]
                depth_pygame = pygame.surfarray.make_surface(depth_img.swapaxes(0, 1))
    
    color_pygame = pygame.surfarray.make_surface(color_img.swapaxes(0, 1))
    camera_img_window.blit(color_pygame, (0, 0)) 
    camera_img_window.blit(depth_pygame, (resolution[0], 0))
    pygame.display.update()  

    episode_idx = env.episode_id

    return recording_flag, episode_idx, color_img, color_img_right

@click.command()
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_config', '-rc', required=True, help='Path to robot_config yaml file')
@click.option('--frequency', '-f', default=20, type=float, help="Control frequency in Hz.")
@click.option('--teleop_command_latency', '-cl', default=0.01, type=float,
              help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
@click.option('--resize_observation_resolution', '-ror', default="1280x720", type=str)
@click.option('--observation_resolution', '-or', default="640x480", type=str)
@click.option('--camera_exposure', '-ce', default=None, type=int)
@click.option('--camera_obs_horizon', '-ch', default=2, type=int)
@click.option('--robot_obs_horizon', '-rh', default=2, type=int)
@click.option('--gripper_obs_horizon', '-gh', default=2, type=int)
@click.option('--active_vision_obs_horizon', '-ah', default=2, type=int)
@click.option('--action_smooth_weight', '-asw', default=0.8, type=int)
@click.option('--bimanual', '-b', is_flag=True, default=False)
@click.option('--verbose', is_flag=True, default=False)
def main(output, robot_config,
         frequency, teleop_command_latency,
         resize_observation_resolution, observation_resolution, camera_exposure,
         camera_obs_horizon, robot_obs_horizon, gripper_obs_horizon, active_vision_obs_horizon,
         action_smooth_weight,
         bimanual,
         verbose
         ):

    global record_condition, base_cam_hand_calib, tx_tip_flange, tx_flange_tip, base_cam_hand_calib_rot, base_cam_hand_calib_trans, action_smooth_dict
    os.makedirs(output, exist_ok=True)

    # load robot config file
    robot_config_data = yaml.safe_load(open(os.path.expanduser(robot_config), 'r'))

    robots_config = robot_config_data['robots']
    recording_flag = record_condition[0]
    episode_idx = 0

    # load robot config file
    robot_config_data = yaml.safe_load(open(os.path.expanduser(robot_config), 'r'))

    robots_config = robot_config_data['robots']
    grippers_config = robot_config_data['grippers']
    camera_config = robot_config_data['cameras']
    active_vision_config = robot_config_data['active_visions']


    # setup experiment
    dt = 1 / frequency

    # obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    resize_res = tuple(int(x) for x in resize_observation_resolution.split('x'))
    obs_res = tuple(int(x) for x in observation_resolution.split('x'))
    # 初始化 Pygame
    pygame.init()
    camera_img_window = pygame.display.set_mode((obs_res[0] * 2, obs_res[1]))  # 第二个窗口  # 78
    pygame.display.set_caption("teleop")

    # load fisheye converter


    # FIXME: 
    cam_scale_coef = 1.25
    human_robot_scale = np.array([1.0, 1.0, 1.0])  # x -5 y +45 z +75
    human_robot_offset = np.array([0.0, 0.0, 0.0])  # x -5 y +45 z +75

    


    teleoperator = VuerTeleop('./real/teleop/inspire_hand_0_4_6_tim.yml', resolution=(obs_res[1], obs_res[0]), distance_to_eye=cam_scale_coef)

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
                    obs_image_resolution=obs_res,  # resolution --(resize)--> resize_res --(crop)--> obs_image_resolution
                    camera_exposure=camera_exposure,
                    down_sample_steps=1,
                    # obs
                    camera_obs_horizon=camera_obs_horizon,
                    robot_obs_horizon=robot_obs_horizon,
                    gripper_obs_horizon=gripper_obs_horizon,
                    active_vision_obs_horizon=active_vision_obs_horizon,
                    # action
                    init_joints=True,
                    # vis params, we vis the resut in the main loop
                    enable_multi_cam_vis=False,
                    verbose=verbose,
                    shm_manager=shm_manager) as env:

            cv2.setNumThreads(2)
            print("Waiting for camera")
            
            time.sleep(2.0)

            print('Ready!')


            while True:
                # ========= human control loop ==========
                print("Human in control!")
                t_start = time.monotonic()
                iter_idx = 0

                while True:
                    # calculate timing
                    t_fps_measure_start = time.time()
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - teleop_command_latency
                    t_command_target = t_cycle_end + dt


                    # pump obs
                    obs = env.get_obs()
                    #################################################################################

                    head_mat, left_pose, right_pose, left_qpos, right_qpos= teleoperator.step()
                    
                    ori_head_mat, ori_left_wrist_mat, ori_right_wrist_mat, ori_left_hand_mat, ori_right_hand_mat = teleoperator.processor.process(
                        teleoperator.tv)

                    # ori_head_mat: head-pose based on world-frame ----> directly map to atv-rot based on camera-base
                    # ori_left_wrist_mat: wrist-rot based on world-frame ----> directly map to wrist-rot based on camera-base
                    # ori_left_wrist_mat: wrist-pos based on head-pose ---->  directly map to wrist-pos based on camera-base
                    # conclusion: head <-----> camera base, where camera-base is the basic pose of camera during eye-hand calibration.

                    # tele-operation control loop
                    recording_flag, episode_idx, color_img, color_img_right = \
                        teleop_state_control(obs, env, recording_flag, episode_idx, ori_left_hand_mat, ori_right_hand_mat, ori_left_wrist_mat, ori_right_wrist_mat, human_robot_offset,
                                             camera_img_window, obs_res, verbose=verbose)
                    
                    
                    if color_img_right is None:
                        img = np.hstack((color_img, color_img))
                    else:
                        img = np.hstack((color_img, color_img_right))
                    # tv_h, tv_w = img.shape[:2]
                    # img = cv2.resize(img, (int(tv_w / cam_scale_coef), int(tv_h / cam_scale_coef)))
                    np.copyto(teleoperator.img_array, img)    # vision streaming
                    # continue


                    # print(f"right_qpos: {right_qpos}")
                    if np.sum(head_mat) == 0:
                        head_mat = np.eye(3)
                    head_rot = rotations.quaternion_from_matrix(head_mat[0:3, 0:3])
                    ypr = rotations.euler_from_quaternion(head_rot, 2, 1, 0, False)
                    # print(f"active_vision_pose is {[0., 0., 0., 0., ypr[0], ypr[1]]}")

                    active_vision_pose = np.array([0, 0, 0, 0] + list(ypr[:2]))
                    active_vision_pose = np.clip(active_vision_pose, -np.pi * 9 / 20, np.pi * 9 / 20)

                    if active_vision_pose[-1] <= -np.pi / 9:
                        active_vision_pose[-1] = -np.pi / 9
                    if active_vision_pose[-1] >= np.pi * 2 / 3:
                        active_vision_pose[-1] = np.pi * 2 / 3

                    # hand_pose_right = [right_qpos[4], right_qpos[6], right_qpos[2], right_qpos[0], right_qpos[9]*10, right_qpos[8]]
                    # hand_pose_left = [left_qpos[4], left_qpos[6], left_qpos[2], left_qpos[0], left_qpos[9]*10, left_qpos[8]]

                    hand_pose_right = [right_qpos[4], right_qpos[6], right_qpos[2], right_qpos[0], right_qpos[9], right_qpos[8]]
                    hand_pose_left = [left_qpos[4], left_qpos[6], left_qpos[2], left_qpos[0], left_qpos[9], left_qpos[8]]
                    

                    ori_right_wrist_mat[:3, -1] = (ori_right_wrist_mat[:3, -1]) * human_robot_scale + human_robot_offset

                    right_wrist_rot = ori_right_wrist_mat[:3, :3] @ base_cam_hand_calib_rot
                    right_wrist_trans = ori_right_wrist_mat[:3, 3] + base_cam_hand_calib_trans
                    right_wrist_mat = np.eye(4)
                    right_wrist_mat[:3, :3] = right_wrist_rot
                    right_wrist_mat[:3, 3] = right_wrist_trans
                    right_wrist_pose = mat_to_pose(right_wrist_mat) 

                    action_dict = dict()
                    if len(env.robots) > 0:
                        action_dict["robot"] = [right_wrist_pose[None]]
                    if len(env.grippers) > 0:
                        action_dict["gripper"] = [np.clip(hand_pose_right, 0, 1000)[None]]
                    if len(env.active_visions) > 0:
                        action_dict["active_vision"] = [active_vision_pose[None]]

                    # if right_gesture > 1:   # not free grasp:
                    #     action_gestures = [np.array([right_gesture])]
                    # else:
                    #     action_gestures = [np.array([-1])]
                    # action_gestures = [np.array([-1])]


                    if bimanual is True:   # right ---> left
                        # action_dict['robot'].append(XXXXXX)
                        action_dict['gripper'].append(np.clip(hand_pose_left, 0, 1000)[None])
                    
                    if action_smooth_dict is None:
                        action_smooth_dict = action_dict
                    else:
                        for key in action_dict.keys():
                            action_dict[key] = action_smooth_weight * action_smooth_dict[key] + (1 - action_smooth_weight) * action_dict[key]
                    
                    precise_wait(t_sample)
                    # execute teleop command
                    if recording_flag == record_condition[1]: # TODO:

                        env.exec_actions(
                            robot_actions=action_dict,
                            timestamps=np.array([t_command_target - time.monotonic() + time.time()]),
                            # gestures=action_gestures,
                            compensate_latency=False
                        )

                    action_smooth_dict = action_dict.copy()
                            
                    precise_wait(t_cycle_end)
                    iter_idx += 1
                    if verbose:
                        print(f"Teleop Mainloop Real FPS: {1 / (time.time() - t_fps_measure_start)}")

# %%
if __name__ == '__main__':
    main()

# python -m tests.test_inspire_hand_agent
# python teleop_real_franka.py -o data/data_robot_raw --robot_config=real/config/franka_inspire_atv_cam_unimanual.yaml -f 20 -ror 1280x720 -or 640x480
