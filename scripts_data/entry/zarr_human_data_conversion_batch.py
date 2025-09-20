import json
from builtins import int
import pickle
import numpy as np 
import cv2 
import pdb
from typing import Sequence, Tuple, Dict, Optional, Union, Generator
from multiprocessing import Process
import os
import pathlib
import click
import imageio
import shutil
import fpsample
from tqdm import tqdm
from common.cv2_util import get_image_transform_resize_crop, intrinsic_transform_resize
from common.cv_util import back_projection
from common.timestamp_accumulator import get_accumulate_timestamp_idxs
from common.replay_buffer import ReplayBuffer
from common.svo_utils import SVOReader
from common.pose_util import euler_pose_to_mat, mat_to_pose, mat_to_euler_pose, pose_to_mat
from common.interpolation_util import PoseInterpolator, get_interp1d
from human_data.constants import yfxrzu2standard

from human_data.hand_retargeting import Hand_Retargeting
hand_retargeting = Hand_Retargeting("./real/teleop/inspire_hand_0_4_6.yml")


def get_eef_pos_velocity(eef_pos_seq):
    delta = np.linalg.norm(eef_pos_seq[1:] - eef_pos_seq[:-1], axis=-1)
    vel = delta.mean()
    return vel


def fast_mat_inv(mat):
    ret = np.eye(4)
    ret[:3, :3] = mat[:3, :3].T
    ret[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
    return ret


def conversion_single_trajectory(
    mode,
    save_dir, 
    calib_quest2camera,
    speed_downsample_ratio,
    single_arm,
    hand_shrink_coef,
    out_resolutions_resize: Union[None, tuple, Dict[str,tuple]]=None, # (width, height)
    out_resolutions_crop: Union[None, tuple, Dict[str,tuple]]=None, # (width, height)
    out_resolutions_image_final: Union[None, tuple, Dict[str,tuple]]=None, # (width, height)
    network_delay_checking: float=1.0,
    num_points_final: int = 2048, 
    points_max_distance_final: float = 1.25,
    ):

    episode_path = os.path.join(save_dir, "episode.pkl")
    if not os.path.exists(episode_path):
        print(f"[Warning] No episode.pkl found in {save_dir}")
        return None
    with open(episode_path, "rb") as f:
        episode_org = pickle.load(f)

    # remove the timestamp where robot0_eef_pos does not change, 6:9 is the wrist pos
    start_time_idx = 0
    eps = 0.005
    for i in range(1, len(episode_org['right_hand_mat'])):
        if np.linalg.norm(episode_org['right_hand_mat'][i, 6:9] - episode_org['right_hand_mat'][0, 6:9]) > eps:
            start_time_idx = i
            break
    for key in episode_org.keys():
        episode_org[key] = episode_org[key][start_time_idx:]

    if speed_downsample_ratio is not None:
        downsample_ratio = speed_downsample_ratio
    else:
        downsample_ratio = 1.0

    dt = (episode_org['timestamp'][1:] - episode_org['timestamp'][:-1]).mean() / downsample_ratio

    print(f"speed_downsample_ratio: {downsample_ratio}")
    print("!" * 25)

    t_init = 4                           # initial 4 frames to be removed
    for key in episode_org.keys():
        episode_org[key] = episode_org[key][t_init:] 

    dt_check = episode_org['timestamp'][1:] - episode_org['timestamp'][:-1]
    dt_check_max = np.max(dt_check)
    if dt_check_max > network_delay_checking:
        print(f"[Warning] Max delay {dt_check_max} > {network_delay_checking}, abandon this episode. {save_dir}")
        return None

    T_start = episode_org['timestamp'][0]
    T_end = episode_org['timestamp'][-1]
    n_steps = int((T_end - T_start) / dt)
    timestamps = np.arange(n_steps + 1) * dt + T_start
    len_hand_arr = len(episode_org['left_hand_mat'][0])
    actions = np.concatenate([episode_org['left_hand_mat'], episode_org['right_hand_mat'], episode_org['head_pose_mat']], axis=1)
    n_pose = actions.shape[1] // 6
    actions_record = []
    for i in range(n_pose):
        act_rotvec = mat_to_pose(euler_pose_to_mat(actions[:, i*6:(i+1)*6]))
        human_pose_interpolator = PoseInterpolator(t=episode_org['timestamp'], x=act_rotvec)
        act_euler = mat_to_euler_pose(pose_to_mat(human_pose_interpolator(timestamps)))
        actions_record.append(act_euler)
    actions_record = np.concatenate(actions_record, axis=1)
    episode_org['left_hand_mat'] = actions_record[:, :len_hand_arr]
    episode_org['right_hand_mat'] = actions_record[:, len_hand_arr:2*len_hand_arr]
    episode_org['head_pose_mat'] = actions_record[:, 2*len_hand_arr:]

    # import pdb; pdb.set_trace()

    # ========================== Transformation to Egocentric View (By Default T=0 Camera View) =================================
    episode_org['head_pose_mat'] = euler_pose_to_mat(episode_org['head_pose_mat']) @ yfxrzu2standard
    # VR2Camera0 = VR --> Quest[0] --> Camera[0]
    vr2camera0 = calib_quest2camera @ fast_mat_inv(episode_org['head_pose_mat'][0])
    # Camera2Camera0 = Camera --> Quest --> VR --> Camera0
    camera_pose = vr2camera0 @ episode_org['head_pose_mat'] @ fast_mat_inv(calib_quest2camera)

    b_hand_joint = len(episode_org['left_hand_mat'][0]) // 6
    left_hand_pose = []
    right_hand_pose = []
    left_hand_pose_rotvec = []
    right_hand_pose_rotvec = []
    for i in range(b_hand_joint):
        # # Hand2Camera0 = Hand --> VR --> Camera0
        left_pose = vr2camera0 @ (euler_pose_to_mat(episode_org['left_hand_mat'][:, i*6:(i+1)*6]) @ yfxrzu2standard)
        right_pose = vr2camera0 @ (euler_pose_to_mat(episode_org['right_hand_mat'][:, i*6:(i+1)*6]) @ yfxrzu2standard)
        left_hand_pose.append(mat_to_euler_pose(left_pose))
        right_hand_pose.append(mat_to_euler_pose(right_pose))
        left_hand_pose_rotvec.append(mat_to_pose(left_pose))
        right_hand_pose_rotvec.append(mat_to_pose(right_pose))

    left_hand_pose = np.concatenate(left_hand_pose, axis=1)
    right_hand_pose = np.concatenate(right_hand_pose, axis=1)
    left_hand_pose_rotvec = np.concatenate(left_hand_pose_rotvec, axis=-1)
    right_hand_pose_rotvec = np.concatenate(right_hand_pose_rotvec, axis=-1)

    # ========================== Hand Retargeting =================================

    """
        Output:
            - left/right_wrist_results:  (T, 6),    wrist 6dof poses in T0-Camera-Coordinate-Coordinate
            - left/right_qpos_results:   (T, 6),    inspire_hand 6dof servo-pos  (pinky, ring, middle, index, thumb-curve, thumb-inside)
            - left/right_opos_results:   (T, 5, 6), original hand 6dof poses in T0-Camera-Coordinate  (thumb, index, middle, ring, pinky)
    """
    left_hand_wrists, right_hand_wrists, left_hand_fix_wrists, right_hand_fix_wrists, left_hand_qposes, right_hand_qposes, left_hand_urdf_qposes, right_hand_urdf_qposes, left_org_hand_poses, right_org_hand_poses = \
        hand_retargeting.retarget(left_hand_pose, right_hand_pose)

    # ========================== Transfer from Euler to RotVec to adapt to Diffusion Policy Controller ==================
    left_hand_wrists = mat_to_pose(euler_pose_to_mat(left_hand_wrists))
    right_hand_wrists = mat_to_pose(euler_pose_to_mat(right_hand_wrists))
    left_hand_fix_wrists = mat_to_pose(euler_pose_to_mat(left_hand_fix_wrists))
    right_hand_fix_wrists = mat_to_pose(euler_pose_to_mat(right_hand_fix_wrists))
    left_org_hand_poses = mat_to_pose(euler_pose_to_mat(left_org_hand_poses.reshape(-1, 6))).reshape(-1, 5, 6)
    right_org_hand_poses = mat_to_pose(euler_pose_to_mat(right_org_hand_poses.reshape(-1, 6))).reshape(-1, 5, 6)


    episode = dict()
    episode['timestamp'] = timestamps
    episode_length = len(episode['timestamp'])
    # episode['hint'] = np.array(["All poses are in T0-Camera-Coordinate"])
    episode['left_hand_pose'] = left_hand_pose_rotvec
    episode['right_hand_pose'] = right_hand_pose_rotvec
    episode['left_wrist_pose'] = left_hand_wrists
    episode['right_wrist_pose'] = right_hand_wrists
    episode['left_wrist_fix_pose'] = left_hand_fix_wrists
    episode['right_wrist_fix_pose'] = right_hand_fix_wrists
    episode['left_finger_pose'] = left_org_hand_poses
    episode['right_finger_pose'] = right_org_hand_poses
    episode['camera0_pose'] = mat_to_pose(camera_pose)
    episode['robot0_eef_pos'] = right_hand_fix_wrists[:, :3]
    episode['robot0_eef_rot_axis_angle'] = right_hand_fix_wrists[:, 3:]
    if single_arm is False:
        episode['robot1_eef_pos'] = left_hand_fix_wrists[:, :3]
        episode['robot1_eef_rot_axis_angle'] = left_hand_fix_wrists[:, 3:]

    if hand_shrink_coef is not None and hand_shrink_coef != 1.0:
        right_hand_qposes_delta = (right_hand_qposes[1:] - right_hand_qposes[:-1]) * hand_shrink_coef
        left_hand_qposes_delta = (left_hand_qposes[1:] - left_hand_qposes[:-1]) * hand_shrink_coef
        right_hand_urdf_qposes_delta = (right_hand_urdf_qposes[1:] - right_hand_urdf_qposes[:-1]) * hand_shrink_coef
        left_hand_urdf_qposes_delta = (left_hand_urdf_qposes[1:] - left_hand_urdf_qposes[:-1]) * hand_shrink_coef
        right_hand_qposes = np.concatenate([np.ones((1, right_hand_qposes.shape[1])) * right_hand_qposes[0:1], right_hand_qposes_delta]).cumsum(axis=0)
        left_hand_qposes = np.concatenate([np.ones((1, left_hand_qposes.shape[1])) * left_hand_qposes[0:1], left_hand_qposes_delta]).cumsum(axis=0)
        right_hand_urdf_qposes = np.concatenate([np.ones((1, right_hand_urdf_qposes.shape[1])) * right_hand_urdf_qposes[0:1], right_hand_urdf_qposes_delta]).cumsum(axis=0)
        left_hand_urdf_qposes = np.concatenate([np.ones((1, left_hand_urdf_qposes.shape[1])) * left_hand_urdf_qposes[0:1], left_hand_urdf_qposes_delta]).cumsum(axis=0)

    episode['gripper0_gripper_pose'] = right_hand_qposes
    episode['urdf_gripper0_gripper_pose'] = right_hand_urdf_qposes

    if single_arm is False:
        episode['gripper1_gripper_pose'] = left_hand_qposes
        episode['urdf_gripper1_gripper_pose'] = left_hand_urdf_qposes

    # ========================== Transformation to Egocentric View (By Default T=0 Head View) =================================

    svo_path = os.path.join(save_dir, "recording.svo2")
    svo_stereo, svo_depth, svo_pointcloud = False, False, False
    if mode in ['d', 'a']:
        svo_depth = True
    if mode in ['s', 'a']:
        svo_stereo = True
    if mode in ['p', 'a']:
        svo_pointcloud = True
    
    with open(os.path.join(save_dir, "device_id.txt"), "r") as f:
        serial_id = f.read().strip()
    svo_camera = SVOReader(svo_path, serial_number=serial_id)
    svo_camera.set_reading_parameters(image=True, depth=svo_depth, pointcloud=svo_pointcloud, concatenate_images=False)
    frame_count = svo_camera.get_frame_count()
    width, height = svo_camera.get_frame_resolution()

    camera_info = svo_camera.get_camera_information()
    # print(f"Camera Information: {(width, height)}, {out_resolutions_resize}, {out_resolutions_crop}, {out_resolutions_image_final}")
    camera_info['left_intrinsic'] = intrinsic_transform_resize(camera_info['left_intrinsic'], input_res=(width, height), output_resize_res=out_resolutions_resize, output_crop_res=out_resolutions_crop)
    camera_info['right_intrinsic'] = intrinsic_transform_resize(camera_info['right_intrinsic'], input_res=(width, height), output_resize_res=out_resolutions_resize, output_crop_res=out_resolutions_crop)
    camera_info['left_intrinsic_final'] = intrinsic_transform_resize(camera_info['left_intrinsic'], input_res=out_resolutions_crop, output_resize_res=out_resolutions_image_final, output_crop_res=out_resolutions_image_final)
    camera_info['right_intrinsic_final'] = intrinsic_transform_resize(camera_info['right_intrinsic'], input_res=out_resolutions_crop, output_resize_res=out_resolutions_image_final, output_crop_res=out_resolutions_image_final)
    camera_info_key = ['stereo_transform', "left_intrinsic", "right_intrinsic", "left_intrinsic_final", "right_intrinsic_final"]
    for key in camera_info_key:
        episode["camera0_" + key] = np.array([camera_info[key]] * episode_length)

    next_global_idx = 0
    
    obs_dict = dict()
    episode['camera0_real_timestamp'] = np.zeros((episode_length,), dtype=np.float64)
    transform_img = get_image_transform_resize_crop(input_res=(width, height), output_resize_res=out_resolutions_resize, output_crop_res=out_resolutions_crop, bgr_to_rgb=True)
    obs_dict['rgb'] = ('image', f'{serial_id}_left', transform_img)
    if svo_stereo:
        obs_dict['rgb_right'] = ('image', f'{serial_id}_right', transform_img)
    if svo_depth:
        transform_depth = get_image_transform_resize_crop(input_res=(width, height), output_resize_res=out_resolutions_resize, output_crop_res=out_resolutions_crop, is_depth=True)
        obs_dict['depth'] = ('depth', f'{serial_id}_left', transform_depth)
        if svo_stereo:
            obs_dict['depth_right'] = ('depth', f'{serial_id}_right', transform_depth)
    if svo_pointcloud:
        transform_pointcloud = get_image_transform_resize_crop(input_res=(width, height), output_resize_res=out_resolutions_resize, output_crop_res=out_resolutions_crop, is_depth=True)
        obs_dict['pointcloud'] = ('pointcloud', f'{serial_id}_left', transform_pointcloud)
    start_time = episode['timestamp'][0]

    frame_cut_fp = os.path.join(save_dir, "frame_cut.txt")
    frame_cut = None
    if os.path.exists(frame_cut_fp):
        # read the number in txt
        with open(frame_cut_fp, "r") as f:
            frame_cut = f.read().strip()
        if frame_cut.isdigit():
            frame_cut = int((episode_org['timestamp'][int(frame_cut) - start_time_idx] - T_start) / dt)
        else:
            print(f"Frame cut {frame_cut} is not a digit, set to None.")
            frame_cut = None 
    if frame_cut is not None:
        episode_length = min(episode_length, frame_cut)
    for episode_key in episode.keys():
        episode[episode_key] = episode[episode_key][:episode_length]

    global_idx = 0
        
    for t in range(frame_count):
        # print(f"{t}: {next_global_idx}")
        svo_output = svo_camera.read_camera(return_timestamp=True)
        if svo_output is None:
            break
        else:
            data_dict, timestamp = svo_output
            timestamp = timestamp / 1000.0
        if timestamp < episode['timestamp'][0] - dt:
            continue

        local_idxs, global_idxs, next_global_idx \
                = get_accumulate_timestamp_idxs(
                timestamps=[timestamp],
                start_time=start_time,
                dt=dt,
                next_global_idx=next_global_idx
            )

        if len(global_idxs) > 0:
            for global_idx in global_idxs:
                if global_idx == episode_length:
                    break
                for key in obs_dict.keys():
                    value = data_dict[obs_dict[key][0]][obs_dict[key][1]]
                    transform = obs_dict[key][2]
                    if value.shape[-1] == 4:
                        value = value[..., :3]
                    value = transform(value)
                    if 'rgb' in key:
                        value = cv2.resize(value, out_resolutions_image_final, interpolation=cv2.INTER_LINEAR)
                    if 'pointcloud' in key:
                        points_xyz = value.reshape(-1, 3)
                        points_rgb = obs_dict['rgb'][2](data_dict['image'][obs_dict[key][1]][..., :3]).reshape(-1, 3)
                        points = np.concatenate([points_xyz, points_rgb / 255.0], axis=-1)
                        # remove NaN point and distance > points_max_distance_final
                        valid_mask = np.linalg.norm(points_xyz, axis=-1) <= points_max_distance_final
                        # valid_mask = valid_mask & np.isfinite(points).all(axis=-1) & (~ np.isnan(points).any(axis=-1))
                        points = points[valid_mask]
                        points_xyz = points_xyz[valid_mask]
                        # np.save("points_org.npy", points)
                        if len(points) > num_points_final:
                            # do furthest point sampling, resulting num_points_final points
                            points_idx = fpsample.bucket_fps_kdline_sampling(points_xyz, num_points_final, h=7)
                        else:
                            # repeat points in points to num_points_final
                            points_idx = np.array([i % len(points_xyz) for i in range(num_points_final)])
                        value = points[points_idx]
                        # np.save("points_ds.npy", value)
                    if 'camera0_' + key not in episode.keys():
                        episode['camera0_' + key] = np.zeros((episode_length,) + value.shape, dtype=value.dtype)
                    episode['camera0_' + key][global_idx] = value
                episode['camera0_real_timestamp'][global_idx] = timestamp
        if (next_global_idx == episode_length) or (global_idx == episode_length):
            break
        
    if (next_global_idx < episode_length) and (global_idx != episode_length):
        abandoned_frames = episode_length - next_global_idx
        for key in episode.keys():
            try:
                episode[key] = episode[key][:-abandoned_frames]
            except:
                pass
        print(f"Warning: {next_global_idx} < {episode_length}, abandoned {abandoned_frames} frames.")

    n_length = np.min([episode['timestamp'].shape[-1], episode['camera0_real_timestamp'].shape[-1]])
    for key in episode.keys():
        episode[key] = episode[key][:n_length]
    print(f"length: {n_length}")

    frame_grasp_fp = os.path.join(save_dir, "frame_grasp.txt")
    frame_grasp = None
    if os.path.exists(frame_grasp_fp):
        # read the number in txt
        with open(frame_grasp_fp, "r") as f:
            frame_grasp = f.read().strip()
        if frame_grasp.isdigit():
            frame_grasp = int((episode_org['timestamp'][int(frame_grasp) - start_time_idx] - T_start) / dt)
        else:
            print(f"Frame frame_grasp {frame_grasp} is not a digit, set to None.")
            frame_grasp = None 
    frame_release_fp = os.path.join(save_dir, "frame_release.txt")
    frame_release = None
    if os.path.exists(frame_release_fp):
        # read the number in txt
        with open(frame_release_fp, "r") as f:
            frame_release = f.read().strip()
        if frame_release.isdigit():
            frame_release = int((episode_org['timestamp'][int(frame_release) - start_time_idx] - T_start) / dt)
        else:
            print(f"Frame frame_release {frame_release} is not a digit, set to None.")
            frame_release = None

    print(f"frame_grasp: {frame_grasp}, frame_release: {frame_release}")
    if frame_grasp is not None:
        assert hand_shrink_coef == 1.0, "Can not set hand_shrink_coef != 1.0 when use frame_grasp and frame_release handler."
        if frame_release is None:
            frame_release = n_length
        gripper0_max_pose = np.max(episode['gripper0_gripper_pose'][frame_grasp:frame_release+1], axis=0)
        if single_arm is False:
            gripper1_max_pose = np.max(episode['gripper1_gripper_pose'][frame_grasp:frame_release+1], axis=0)
        assert frame_grasp >= 10 + 1
        episode['gripper0_gripper_pose'][frame_grasp:frame_release+1] = gripper0_max_pose[None]
        if single_arm is False:
            episode['gripper1_gripper_pose'][frame_grasp:frame_release+1] = gripper1_max_pose[None]
        gripper_interp = get_interp1d(episode['timestamp'][[frame_grasp-10-1, frame_grasp]], episode['gripper0_gripper_pose'][[frame_grasp-10-1,frame_grasp]])
        episode['gripper0_gripper_pose'][frame_grasp-10:frame_grasp] = gripper_interp(episode['timestamp'][frame_grasp-10:frame_grasp])
        if single_arm is False:
            gripper_interp = get_interp1d(episode['timestamp'][[frame_grasp-10-1, frame_grasp]], episode['gripper1_gripper_pose'][[frame_grasp-10-1,frame_grasp]])
            episode['gripper1_gripper_pose'][frame_grasp-10:frame_grasp] = gripper_interp(episode['timestamp'][frame_grasp-10:frame_grasp])
        if frame_release < n_length:
            frame_release_finish = min(n_length, frame_release + 10) 
            gripper_interp = get_interp1d(episode['timestamp'][[frame_release, frame_release_finish]], episode['gripper0_gripper_pose'][[frame_release, frame_release_finish]])
            episode['gripper0_gripper_pose'][frame_release+1: frame_release_finish] = gripper_interp(episode['timestamp'][frame_release+1: frame_release_finish])
            if single_arm is False:
                gripper_interp = get_interp1d(episode['timestamp'][[frame_release, frame_release_finish]], episode['gripper1_gripper_pose'][[frame_release, frame_release_finish]])
                episode['gripper1_gripper_pose'][frame_release+1: frame_release_finish] = gripper_interp(episode['timestamp'][frame_release+1: frame_release_finish])

    episode['action'] = np.concatenate([
        episode['robot0_eef_pos'], episode['robot0_eef_rot_axis_angle'], episode['gripper0_gripper_pose'],
    ], axis=-1)

    if single_arm is False:
        episode['action'] = np.concatenate([
            episode['action'], episode['robot1_eef_pos'], episode['robot1_eef_rot_axis_angle'], episode['gripper1_gripper_pose'],
        ], axis=-1)
    
    # # save episode['camera0_rgb_right'] as test.mp4
    # if 'camera0_rgb_right' in episode.keys():
    #     rgb_right = episode['camera0_rgb_right']
    #     if rgb_right.shape[0] > 0:
    #         video_writer = imageio.get_writer("test.mp4", fps=fps)
    #         for i in range(rgb_right.shape[0]):
    #             video_writer.append_data(rgb_right[i])
    #         video_writer.close()
    #     else:
    #         print("No right camera data found, skip saving test.mp4.") 
        
    # for key in episode.keys():
    #     try:
    #         print(f"Key: {key}, shape: {episode[key].shape}")
    #     except:
    #         print(f"Key: {key}, {episode[key]}")
    
    return episode


def conversion_trajectory(input_data_fp_list, calib_quest2camera, speed_downsample_ratio, single_arm,
                          hand_shrink_coef, 
                          mode, 
                          out_resolutions_resize, out_resolutions_crop, resolution_image_final, num_points_final, points_max_distance_final,
                          replay_buffer, 
                          network_delay_checking,
                          process_id):
    pbar = tqdm(enumerate(input_data_fp_list), desc=f"Process {process_id}")
    for i, input_data_fp in pbar:
        save_dir, source, source_idx = input_data_fp
        episode = conversion_single_trajectory(
            mode,
            save_dir, 
            calib_quest2camera,
            speed_downsample_ratio,
            single_arm,
            hand_shrink_coef,
            out_resolutions_resize,
            out_resolutions_crop,
            resolution_image_final,
            network_delay_checking,
            num_points_final, points_max_distance_final,
        )
        if episode is None:
            pbar.write(f"Skip {save_dir} due to network delay or no data.")
            continue
        episode['embodiment'] = np.zeros((len(episode['robot0_eef_pos']), 1))
        episode['source_idx'] = np.ones((len(episode['robot0_eef_pos']), 1)) * source_idx
        if episode is not None:
            replay_buffer.add_episode(episode, compressors='disk')  # with lock mechanism inside replay_buffer instance



@click.command()
@click.option('--input_dir', '-i',  required=True)
@click.option('--output', '-o', required=True)
@click.option('--calib_quest2camera_file', '-cf', required=True)
@click.option('--adapt_config_file', '-acf', default=None, type=str)
@click.option('--single_arm', '-cf', is_flag=True, default=False)
@click.option('--default_speed_downsample_ratio', '-dsdr', default=1.0, type=float)
@click.option('--default_hand_shrink_coef', '-dhsc', default=1.0, type=float)
@click.option('--mode', '-m', required=True, type=click.Choice(['o', 'p', 's', 'a'], case_sensitive=False), default='o',
    help="o: only image, p: with pointcloud, s: with stereo-image, a: with all, including pointcloud and stereo")
@click.option('--resolution_resize', '-ror', default='640x480')
@click.option('--resolution_crop', '-or', default='640x480')
@click.option('--resolution_image_final', '-for', default='224x224')
@click.option('--num_use_source', '-nus', default=None, type=int)
@click.option('--num_points_final', '-npf', type=int, default=2048)
@click.option('--points_max_distance_final', '-pmdf', type=float, default=1.0)
@click.option('--n_encoding_threads', '-ne', default=-1, type=int)
@click.option('--network_delay_checking', '-dl', default=0.5, help="Max network delay for tolerance.")
def main(input_dir, output, calib_quest2camera_file, 
         adapt_config_file, single_arm,
         default_speed_downsample_ratio, default_hand_shrink_coef,
         mode, 
         resolution_resize, resolution_crop, resolution_image_final, num_use_source, num_points_final, points_max_distance_final,
         n_encoding_threads, 
         network_delay_checking
        ):
    out_resolution_resize = tuple(int(x) for x in resolution_resize.split('x'))
    out_resolution_crop = tuple(int(x) for x in resolution_crop.split('x'))
    resolution_image_final = tuple(int(x) for x in resolution_image_final.split('x'))

    if adapt_config_file is not None:
        # load json file
        with open(adapt_config_file, 'r') as f:
            adapt_config = json.load(f)

    input_dir_list = os.listdir(input_dir)
    input_dir_list = [os.path.join(input_dir, x) for x in input_dir_list]

    for input_dir in input_dir_list:
        input_folder = input_dir.split('/')[-1]
        embodiment = input_folder.split('_')[0]
        assert embodiment == "human"
        environment_setting = input_folder.split('_')[1]
        instruction = '_'.join(input_folder.split('_')[2:])
        
        replay_buffer_fp = os.path.join(output, embodiment + "_" + environment_setting + "+" + instruction + "+")
        
        if environment_setting == "me":  # multi-environment
            if num_use_source is not None and num_use_source < 0:
                num_use_source = None
            if num_use_source is not None:
                replay_buffer_fp = replay_buffer_fp + "_src" + str(num_use_source)
        replay_buffer_fp = replay_buffer_fp + ".zarr"
        replay_buffer_fp = pathlib.Path(os.path.expanduser(replay_buffer_fp))

        if replay_buffer_fp.exists():
            shutil.rmtree(replay_buffer_fp)
        
        replay_buffer = ReplayBuffer.create_from_path(replay_buffer_fp, mode='a')

        if environment_setting == "me":  # multi-environment
            input_data_fp_list = []
            source_list = os.listdir(input_dir)
            source_list.sort()
            for sidx, source in enumerate(source_list):
                if num_use_source is not None and sidx == num_use_source:
                    print(f"Use Source: {source_list[:num_use_source]}")
                    break
                tmp_fp_list = os.listdir(os.path.join(input_dir, source))
                tmp_fp_list = [(os.path.join(input_dir, source, fp), source, sidx) for fp in tmp_fp_list]
                input_data_fp_list = input_data_fp_list + tmp_fp_list
                if sidx + 1 == num_use_source:
                    print(f"Use Source: {source_list[:num_use_source]}")
                    print("Use All Sources.")
                    break
        else:
            input_data_fp_list = os.listdir(input_dir)
            input_data_fp_list.sort()
            input_data_fp_list = [(os.path.join(input_dir, fp), "default", 0) for fp in input_data_fp_list] 
        calib_quest2camera = np.load(calib_quest2camera_file)

        speed_downsample_ratio = None
        hand_shrink_coef = None
        if input_folder in adapt_config:
            adapt_param = adapt_config[input_folder]
            if 'speed_downsample_ratio' in adapt_param:
                speed_downsample_ratio = adapt_param['speed_downsample_ratio']
            else:
                speed_downsample_ratio = default_speed_downsample_ratio
            if 'hand_shrink_coef' in adapt_param:
                hand_shrink_coef = adapt_param['hand_shrink_coef']
            else:
                hand_shrink_coef = default_hand_shrink_coef
        else:
            hand_shrink_coef, speed_downsample_ratio = default_hand_shrink_coef, default_speed_downsample_ratio

        print(f"Input Directory: {input_dir}")
        print(f"Output Directory: {replay_buffer_fp}")
        print(f"Hand Shrink Coefficient: {hand_shrink_coef}")
        print(f"Speed Downsample Refer Zarr: {speed_downsample_ratio}")

        if n_encoding_threads > 1:
            input_data_fp_batch_list = []
            for i in range(n_encoding_threads):
                input_data_fp_batch_list.append(input_data_fp_list[i::n_encoding_threads])

            process_list = []
            for i in range(n_encoding_threads):
                p = Process(target=conversion_trajectory, args=(input_data_fp_batch_list[i], calib_quest2camera, speed_downsample_ratio, single_arm, hand_shrink_coef, mode, out_resolution_resize, out_resolution_crop, resolution_image_final, num_points_final, points_max_distance_final, replay_buffer, network_delay_checking, i))
                p.start()
                process_list.append(p)

            for p in process_list:
                p.join()
        else:
            conversion_trajectory(input_data_fp_list, calib_quest2camera, speed_downsample_ratio, single_arm, hand_shrink_coef, mode, out_resolution_resize, out_resolution_crop, resolution_image_final, num_points_final, points_max_distance_final, replay_buffer, network_delay_checking, 0)
        
        print(f"Saving to disk finish: Task {input_dir}")


if __name__ == "__main__":
    # test_conversion()
    main()