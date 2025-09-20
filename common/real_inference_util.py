from typing import Dict, Callable, Tuple, List
import numpy as np
import collections
from omegaconf import OmegaConf
import fpsample
import cv2
from .cv2_util import get_image_transform
from .pose_repr_util import (
    compute_relative_pose, 
    convert_pose_mat_rep
)
from .pose_util import (
    pose_to_mat, mat_to_pose, 
    mat_to_pose10d, pose10d_to_mat)
from common.cv_util import egocentric_to_base_obs_transformation


def get_real_obs_resolution(
        shape_meta: dict
        ) -> Tuple[int, int]:
    out_res = None
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            co,ho,wo = shape
            if out_res is None:
                out_res = (wo, ho)
            assert out_res == (wo, ho)
    return out_res


def get_real_obs_dict(
        env_obs: Dict[str, np.ndarray], 
        shape_meta: dict,
        ) -> Dict[str, np.ndarray]:
    obs_dict_np = dict()
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            this_imgs_in = env_obs[key]
            t,hi,wi,ci = this_imgs_in.shape
            co,ho,wo = shape
            assert ci == co
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
                tf = get_image_transform(
                    input_res=(wi,hi), 
                    output_res=(wo,ho), 
                    bgr_to_rgb=False)
                out_imgs = np.stack([tf(x) for x in this_imgs_in])
                if this_imgs_in.dtype == np.uint8:
                    out_imgs = out_imgs.astype(np.float32) / 255
            # THWC to TCHW
            obs_dict_np[key] = np.moveaxis(out_imgs,-1,1)
        elif type == 'low_dim':
            this_data_in = env_obs[key]
            obs_dict_np[key] = this_data_in
    return obs_dict_np


def get_real_hra_obs_dict(
        env_obs: Dict[str, np.ndarray], 
        shape_meta: dict,
        obs_pose_repr: str='abs',
        img_obs_shape: tuple=(224, 224),
        calib_cam2base: np.ndarray=None,
        hand_to_eef: np.ndarray=None,
        num_points_final: int=1024,
        points_max_distance_final: float=1.0,
        ) -> Dict[str, np.ndarray]:
    obs_dict_np = dict()

    for key in env_obs.keys():
        if (key.startswith('robot') or key.startswith('gripper')) and (not key.endswith('force')) and (key not in shape_meta['obs']):
            # to_container first
            shape_meta = OmegaConf.to_container(shape_meta, resolve=True)
            shape_meta['obs'][key] = {
                'shape': [env_obs[key].shape[1:][i] for i in range(len(env_obs[key].shape[1:]))],
                'horizon': env_obs[key].shape[0],
                'latency': 0,
                'down_sample_steps': shape_meta['obs']['camera0_rgb']['down_sample_steps'],
                'type': 'low_dim',
                'ignore_by_policy': True,
                'embedding_dim': -1,
            }
            shape_meta = OmegaConf.create(shape_meta)

    # process rgb-shape and base-to-egocentric pose
    obs_shape_meta = shape_meta['obs']
    robot_prefix_map = list()
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        if type == 'rgb':
            if 'rgb' in key:
                this_imgs_in = env_obs[key]
                out_imgs = []
                for i in range(this_imgs_in.shape[0]):
                    this_img_in = this_imgs_in[i]
                    out_img = cv2.resize(this_img_in, img_obs_shape, interpolation=cv2.INTER_LINEAR)
                    out_imgs.append(out_img)
                out_imgs = np.stack(out_imgs, axis=0)
                obs_dict_np[key] = np.moveaxis(out_imgs,-1,1) / 255.0
            elif 'pointcloud' in key:
                pointclouds = []
                n_t = len(env_obs[key])
                for i in range(n_t):
                    points_xyz = env_obs[key][i].reshape(-1, 3) / 1000.0
                    image_key = key.replace('pointcloud', 'rgb')
                    points_rgb = env_obs[image_key].reshape(-1, 3) / 255.0
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
                    pointclouds.append(points[points_idx])
                obs_dict_np[key] = np.stack(pointclouds)

        elif type == 'low_dim' and ('eef_pos' in key):
            robot_prefix = key.replace('_eef_pos', '')
            robot_prefix_map.append(robot_prefix)
            pair_key = key.replace('eef_pos', 'eef_rot_axis_angle')
            gripper_to_base = np.concatenate([env_obs[key], env_obs[pair_key]], axis=-1)

            gripper_to_base = mat_to_pose(pose_to_mat(gripper_to_base) @ hand_to_eef)
            gripper_to_cam = egocentric_to_base_obs_transformation(pose2cam=gripper_to_base, cam2base=calib_cam2base, inv_cam2base=True)
            obs_dict_np[key] = gripper_to_cam[:, :3]
            obs_dict_np[pair_key] = gripper_to_cam[:, 3:]
        elif type == 'low_dim' and ('eef_rot_axis_angle' in key):
            pass 
        elif type == 'low_dim' and ('gripper_pose' in key or 'embodiment' in key or 'task_id' in key or 'gripper_force' in key):
            obs_dict_np[key] = env_obs[key]
        elif type == 'low_dim':
            raise NotImplementedError(f"Unknown type {type} for key {key} in obs")
    
    if 'instruction' in env_obs:
        obs_dict_np['instruction'] = env_obs['instruction']

    # generate relative pose
    for robot_prefix in robot_prefix_map:
        # convert pose to mat
        pose_mat = pose_to_mat(np.concatenate([
            obs_dict_np[robot_prefix + '_eef_pos'],
            obs_dict_np[robot_prefix + '_eef_rot_axis_angle']
        ], axis=-1))
        obs_dict_np[robot_prefix+"_ego_pose_mat"] = pose_mat[-1]

        # solve reltaive obs
        obs_pose_mat = convert_pose_mat_rep(
            pose_mat, 
            base_pose_mat=pose_mat[-1],
            pose_rep=obs_pose_repr,
            backward=False)

        obs_pose = mat_to_pose10d(obs_pose_mat)
        obs_dict_np[robot_prefix + '_eef_pos'] = obs_pose[...,:3]
        obs_dict_np[robot_prefix + '_eef_rot_axis_angle'] = obs_pose[...,3:]

        # gripper_prefix = robot_prefix.replace('robot', 'gripper')
        # gripper_pose = obs_dict_np[gripper_prefix.replace('robot', 'gripper') + '_gripper_pose']
        # obs_dict_np[gripper_prefix+"_gripper_pose_base"] = gripper_pose[-1]
        # if obs_pose_repr == 'relative':
        #     obs_dict_np[robot_prefix + '_gripper_pose'] = gripper_pose - gripper_pose[-1]

    return obs_dict_np 



def get_real_hra_action(
        action: np.ndarray,
        env_obs: Dict[str, np.ndarray], 
        action_pose_repr: str='abs',
        calib_cam2base: np.ndarray=None,
        hand_to_eef: np.ndarray=None,
        n_robots: int=1
    ):

    env_action = list()
    for robot_idx in range(n_robots):
        # convert pose to mat
        pose_mat = env_obs[f'robot{robot_idx}_ego_pose_mat']

        action_pose10d = action[..., robot_idx*15:robot_idx*15+9]
        action_pose_mat = pose10d_to_mat(action_pose10d)

        # import pdb; pdb.set_trace()
        # action_pose_repr = 'abs'

        # solve relative action
        action_mat = convert_pose_mat_rep(
            action_pose_mat, 
            base_pose_mat=pose_mat,
            pose_rep=action_pose_repr,
            backward=True)

        # convert action to pose
        action_pose = mat_to_pose(action_mat)
        hand_to_base = egocentric_to_base_obs_transformation(pose2cam=action_pose, cam2base=calib_cam2base, inv_cam2base=False)

        gripper_to_base = pose_to_mat(hand_to_base) @ np.linalg.inv(hand_to_eef)
        gripper_to_base = mat_to_pose(gripper_to_base)

        env_action.append(gripper_to_base)

        gripper_raw_action = action[..., robot_idx*15+9:(robot_idx+1)*15]
        gripper_action = gripper_raw_action
        env_action.append(gripper_action)

    env_action = np.concatenate(env_action, axis=-1)
    return env_action
    


def get_replay_hra_action(
        action: np.ndarray,
        env_obs: Dict[str, np.ndarray], 
        action_pose_repr: str='abs',
        calib_cam2base: np.ndarray=None,
        hand_to_eef: np.ndarray=None,
        n_robots: int=1,
    ):

    env_action = list()
    for robot_idx in range(n_robots):
        # convert pose to mat
        pose_mat = env_obs[f'robot{robot_idx}_ego_pose_mat']

        action_pose6d = action[..., robot_idx*12:robot_idx*12+6]
        action_pose_mat = pose_to_mat(action_pose6d)

        # solve relative action
        action_mat = convert_pose_mat_rep(
            action_pose_mat, 
            base_pose_mat=pose_mat,
            pose_rep=action_pose_repr,
            backward=True)

        # convert action to pose
        action_pose = mat_to_pose(action_mat)
        hand_to_base = egocentric_to_base_obs_transformation(pose2cam=action_pose, cam2base=calib_cam2base, inv_cam2base=False)

        gripper_to_base = pose_to_mat(hand_to_base) @ np.linalg.inv(hand_to_eef)
        gripper_to_base = mat_to_pose(gripper_to_base)
        # gripper_to_base = hand_to_base

        env_action.append(gripper_to_base)
        env_action.append(action[..., robot_idx*12+6:robot_idx*12+12])
    env_action = np.concatenate(env_action, axis=-1)

    return env_action
    