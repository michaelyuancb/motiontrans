import pickle
import numpy as np
import cv2
import pyzed.sl as sl
from typing import Sequence, Tuple, Dict, Optional, Union, Generator
import os
import pathlib
import yaml
import click
import fpsample
import shutil
from copy import deepcopy
from multiprocessing import Process
from tqdm import tqdm
from common.cv2_util import get_image_transform_resize_crop
from common.cv_util import egocentric_to_base_obs_transformation
from common.timestamp_accumulator import get_accumulate_timestamp_idxs
from common.cv2_util import get_image_transform_resize_crop, intrinsic_transform_resize
from common.replay_buffer import ReplayBuffer
from common.svo_utils import SVOReader
from common.interpolation_util import PoseInterpolator, get_interp1d
from real.teleop.teleop_utils import pose_to_mat, mat_to_pose


def get_eef_pos_velocity(eef_pos_seq):
    delta = np.linalg.norm(eef_pos_seq[1:] - eef_pos_seq[:-1], axis=-1)
    vel = delta.mean()
    return vel



def conversion_single_trajectory(
        hand_to_eef,
        mode,
        save_dir,
        out_resolutions_resize: Union[None, tuple, Dict[str, tuple]] = None,  # (width, height)
        out_resolutions_crop: Union[None, tuple, Dict[str, tuple]] = None,  # (width, height)
        out_resolutions_image_final: Union[None, tuple, Dict[str, tuple]] = None,  # (width, height)
        num_points_final: int = 2048, 
        points_max_distance_final: float = 1.25,
):
    cfg_path = os.path.join(save_dir, "episode_config.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    n_robot = len(cfg['robots'])

    episode_path = os.path.join(save_dir, "episode.pkl")
    with open(episode_path, "rb") as f:
        episode = pickle.load(f)

    # remove the timestamp where robot0_eef_pos does not change
    start_time_idx = 0
    eps = 0.005
    for i in range(1, len(episode[f'robot0_eef_pos'])):
        if np.linalg.norm(episode[f'robot0_eef_pos'][i] - episode[f'robot0_eef_pos'][0]) > eps:
            start_time_idx = i
            break
    for key in episode.keys():
        episode[key] = episode[key][start_time_idx:]

    #################################### Speed Updatation #####################################

    dt = (episode['timestamp'][1:] - episode['timestamp'][:-1]).mean()
    # ========================== Remove Initialization Stage Data =================================

    t_init = 0
    while (episode['action'][t_init] == 0.).all():
        t_init += 1
    for key in episode.keys():
        episode[key] = episode[key][t_init:]
    episode_length = len(episode['timestamp'])

    # ========================== Transformation to Egocentric View =================================
    assert len(cfg['cameras']) == 1  # TODO: support multiple cameras
    n_robot = len(cfg['robots'])
    st_action_dim_idx = 0
    for robot_id in range(n_robot):
        cam_to_base = cfg['cameras'][0]['calib_cam_to_base'][robot_id]
        base_id, cam2base_pose = cam_to_base[0], cam_to_base[1]

        # TODO: currently we only support for single hand calibration !!!!!!
        assert base_id == robot_id
        action_dim = cfg['robots'][robot_id]['action_dim']

        assert action_dim == 6
        gripper_to_base = episode['action'][:, st_action_dim_idx: st_action_dim_idx+action_dim]
        gripper_to_base = mat_to_pose(pose_to_mat(gripper_to_base) @ hand_to_eef)
        gripper_to_cam = egocentric_to_base_obs_transformation(pose2cam=gripper_to_base, cam2base=cam2base_pose,
                                                               inv_cam2base=True)
        episode['action'][:, st_action_dim_idx: st_action_dim_idx+action_dim] = gripper_to_cam

        gripper_to_base = np.concatenate([episode[f'robot{robot_id}_eef_pos'], episode[f'robot{robot_id}_eef_rot_axis_angle']], axis=1)
        gripper_to_base = mat_to_pose(pose_to_mat(gripper_to_base) @ hand_to_eef)
        gripper_to_cam = egocentric_to_base_obs_transformation(pose2cam=gripper_to_base, cam2base=cam2base_pose,
                                                               inv_cam2base=True)
        episode[f'robot{robot_id}_eef_pos'] = gripper_to_cam[:, :3]
        episode[f'robot{robot_id}_eef_rot_axis_angle'] = gripper_to_cam[:, 3:]

        st_action_dim_idx = st_action_dim_idx + action_dim + cfg['grippers'][robot_id]['action_dim']

    # ========================== Action Class Initialization =================================
    ### robot -> gripper, right_hand -> left_hand
    st_action_dim_idx = 0
    actions = []
    for robot_id in range(n_robot):
        action_dim = cfg['robots'][robot_id]['action_dim']
        assert action_dim == 6
        actions.append(episode[f'robot{robot_id}_eef_pos'])
        actions.append(episode[f'robot{robot_id}_eef_rot_axis_angle'])
        st_action_dim_idx += action_dim

        action_dim = cfg['grippers'][robot_id]['action_dim']
        actions.append(episode[f'gripper{robot_id}_gripper_pose'])
        st_action_dim_idx += action_dim

    for i in range(len(actions)):
        if actions[i].ndim == 3:
            actions[i] = actions[i][..., 0]
    actions = np.concatenate(actions, axis=-1)
    episode['action'] = actions

    # ========================== Camera Action Alignment =================================
    svo_path = os.path.join(save_dir, "videos", "recording.svo2")
    svo_stereo, svo_pointcloud = False, False
    if mode in ['p', 'a']:
        svo_pointcloud = True
    if mode in ['s', 'a']:
        svo_stereo = True

    serial_id = str(cfg['cameras'][0]['device_id'])
    svo_camera = SVOReader(svo_path, serial_number=serial_id)
    svo_camera.set_reading_parameters(image=True, depth=False, pointcloud=svo_pointcloud, concatenate_images=False)
    frame_count = svo_camera.get_frame_count()
    width, height = svo_camera.get_frame_resolution()
    try:
        camera_info = svo_camera.get_camera_information()
        camera_info['left_intrinsic'] = intrinsic_transform_resize(camera_info['left_intrinsic'], input_res=(width, height), output_resize_res=out_resolutions_resize, output_crop_res=out_resolutions_crop)
        camera_info['right_intrinsic'] = intrinsic_transform_resize(camera_info['right_intrinsic'], input_res=(width, height), output_resize_res=out_resolutions_resize, output_crop_res=out_resolutions_crop)
        camera_info['left_intrinsic_final'] = intrinsic_transform_resize(camera_info['left_intrinsic'], input_res=out_resolutions_crop, output_resize_res=out_resolutions_image_final, output_crop_res=out_resolutions_image_final)
        camera_info['right_intrinsic_final'] = intrinsic_transform_resize(camera_info['right_intrinsic'], input_res=out_resolutions_crop, output_resize_res=out_resolutions_image_final, output_crop_res=out_resolutions_image_final)
        camera_info_key = ['stereo_transform', "left_intrinsic", "right_intrinsic", "left_intrinsic_final", "right_intrinsic_final"]
    except:
        import pdb; pdb.set_trace()
    for key in camera_info_key:
        episode["camera0_" + key] = np.array([camera_info[key]] * episode_length)

    next_global_idx = 0
    start_time = episode['timestamp'][0]

    obs_dict = dict()
    episode['camera0_real_timestamp'] = np.zeros((episode_length,), dtype=np.float64)
    transform_img = get_image_transform_resize_crop(input_res=(width, height), output_resize_res=out_resolutions_resize,
                                                    output_crop_res=out_resolutions_crop, bgr_to_rgb=True)
    obs_dict['rgb'] = ('image', f'{serial_id}_left', transform_img)
    if svo_stereo:
        obs_dict['rgb_right'] = ('image', f'{serial_id}_right', transform_img)
    if svo_pointcloud:
        transform_pointcloud = get_image_transform_resize_crop(input_res=(width, height),
                                                               output_resize_res=out_resolutions_resize,
                                                               output_crop_res=out_resolutions_crop, is_pcd=True)
        obs_dict['pointcloud'] = ('pointcloud', f'{serial_id}_left', transform_pointcloud)

    frame_cut_fp = os.path.join(save_dir, "frame_cut.txt")
    frame_cut = None
    if os.path.exists(frame_cut_fp):
        # read the number in txt
        with open(frame_cut_fp, "r") as f:
            frame_cut = f.read().strip()
        if frame_cut.isdigit():
            frame_cut = int(frame_cut) - start_time_idx
        else:
            print(f"Frame cut {frame_cut} is not a digit, set to None.")
            frame_cut = None 
    if frame_cut is not None:
        episode_length = min(episode_length, frame_cut)
    for episode_key in episode.keys():
        episode[episode_key] = episode[episode_key][:episode_length]

    global_idx = 0

    # for t in tqdm(range(frame_count)):
    for t in range(frame_count):
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
                if global_idx >= episode_length:
                    print(f"Warning: global_idx {global_idx} >= episode_length {episode_length}, break.")
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
        if next_global_idx == episode_length:
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

    # for key in episode.keys():
    #     try:
    #         print(f"Key: {key}, shape: {episode[key].shape}")
    #     except:
    #         print(f"Key: {key}, {episode[key]}")

    return episode


def conversion_trajectory(input_data_fp_list, hand_to_eef, mode, out_resolutions_resize,
                          out_resolutions_crop, 
                          resolution_image_final,
                          num_points_final, points_max_distance_final,
                          replay_buffer,
                          process_id, verbose):
    if verbose is True:
        pbar = tqdm(input_data_fp_list, desc=f"Process {process_id}")
    else:
        pbar = input_data_fp_list
    for input_data_fp in pbar:
        save_dir, source, source_idx = input_data_fp
        episode = conversion_single_trajectory(
            hand_to_eef,
            mode,
            save_dir,
            out_resolutions_resize,
            out_resolutions_crop,
            resolution_image_final,
            num_points_final, points_max_distance_final,
        )
        episode['embodiment'] = np.ones((len(episode['robot0_eef_pos']), 1))
        episode['source_idx'] = np.ones((len(episode['robot0_eef_pos']), 1)) * source_idx
        replay_buffer.add_episode(episode, compressors='disk')  # with lock mechanism inside replay_buffer instance


@click.command()
@click.option('--input_dir', '-i', required=True)
@click.option('--output', '-o', required=True)
@click.option('--instruction', '-ins', type=str, required=True)
@click.option('--hand_to_eef_file', '-ehf', required=True)
@click.option('--mode', '-m', required=True, type=click.Choice(['o', 'p', 's', 'a'], case_sensitive=False), default='o',
              help="o: only image, p: with pointcloud, s: with stereo-image, a: with all, including pointcloud and stereo")
@click.option('--resolution_resize', '-ror', default='1280x720')
@click.option('--resolution_crop', '-or', default='640x480')
@click.option('--resolution_image_final', '-for', default='224x224')
@click.option('--num_use_source', '-nus', default=None, type=int)
@click.option('--num_points_final', '-npf', type=int, default=2048)
@click.option('--points_max_distance_final', '-pmdf', type=float, default=1.0)
@click.option('--n_encoding_threads', '-ne', default=-1, type=int)
@click.option('--commit', '-c', default="", type=None)
@click.option('--verbose', '-v', is_flag=True, default=True)
def main(input_dir, output, instruction, hand_to_eef_file, mode, 
         resolution_resize, resolution_crop, resolution_image_final, num_use_source, num_points_final, points_max_distance_final,
         n_encoding_threads, commit,
         verbose):
    hand_to_eef = np.load(hand_to_eef_file)
    out_resolution_resize = tuple(int(x) for x in resolution_resize.split('x'))
    out_resolution_crop = tuple(int(x) for x in resolution_crop.split('x'))
    resolution_image_final = tuple(int(x) for x in resolution_image_final.split('x'))
    input_folder = input_dir.split('/')[-1]
    multi_env_setting = input_folder.split('_')[1]
    while len(instruction) > 0 and instruction[-1] == '.':
        instruction = instruction[:-1]
    if len(instruction) > 0:
        pass
    else:
        instruction = '_'.join(input_folder.split('_')[2:])
    replay_buffer_fp = os.path.join(output, input_folder + "_" + resolution_resize + "_" + resolution_crop + "+" + instruction + "+")
    if multi_env_setting == "me":
        if num_use_source is not None and num_use_source < 0:
            num_use_source = None
        if num_use_source is not None:
            replay_buffer_fp = replay_buffer_fp + "_src" + str(num_use_source)
    if len(commit) > 0 and commit != "_":
        replay_buffer_fp += "_" + commit
    replay_buffer_fp = replay_buffer_fp + ".zarr"
    replay_buffer_fp = pathlib.Path(os.path.expanduser(replay_buffer_fp))
    if replay_buffer_fp.exists():
        click.confirm('Output path already exists! Overrite?', abort=True)
    if replay_buffer_fp.exists():
        shutil.rmtree(replay_buffer_fp)

    replay_buffer = ReplayBuffer.create_from_path(replay_buffer_fp, mode='a')

    if multi_env_setting == "me":          # multi-source
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
        tmp = input("continue? (y/n): ")
        if tmp.lower() != 'y':
            exit(0)
    else:
        input_data_fp_list = os.listdir(input_dir)
        input_data_fp_list.sort()
        input_data_fp_list = [(os.path.join(input_dir, fp), "default", 0) for fp in input_data_fp_list] 

    if n_encoding_threads > 1:
        input_data_fp_batch_list = []
        for i in range(n_encoding_threads):
            input_data_fp_batch_list.append(input_data_fp_list[i::n_encoding_threads])

        process_list = []
        for i in range(n_encoding_threads):
            p = Process(target=conversion_trajectory, args=(
            input_data_fp_batch_list[i], hand_to_eef, mode, out_resolution_resize, out_resolution_crop, resolution_image_final,
            num_points_final, points_max_distance_final,
            replay_buffer, i, verbose))
            p.start()
            process_list.append(p)

        for p in process_list:
            p.join()
    else:
        conversion_trajectory(input_data_fp_list, hand_to_eef, mode, out_resolution_resize, out_resolution_crop, resolution_image_final,
                              num_points_final, points_max_distance_final,
                              replay_buffer, 0, verbose)

    print("Saving to disk")


if __name__ == "__main__":
    # test_conversion()
    main()

# bash scripts_data/robot_data_conversion_base.sh