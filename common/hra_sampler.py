from typing import Optional
import omegaconf
import numpy as np
import random
import scipy.interpolate as si
import scipy.spatial.transform as st
from common.replay_buffer import ReplayBuffer
from common.pose_util import pose_to_mat, mat_to_pose

def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes-1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


class HRASequenceSampler:
    """
    Sample sequences from the replay buffer for Human-Robot-Action (HRA) training.
    Very similar to original DiffusionPolicy SequenceSampler, but support camera pose re-projection for current_idx view.
    """

    def __init__(self,
        embodiment: str,
        shape_meta: dict,
        replay_buffer: ReplayBuffer,
        rgb_keys: list,
        lowdim_keys: list,
        key_horizon: dict,
        key_latency_steps: dict,
        key_down_sample_steps: dict,
        robot_action_dim: list,
        gripper_action_dim: list,
        episode_mask: Optional[np.ndarray]=None,
        action_padding_ratio: float=1.0,
        repeat_frame_prob: float=0.0,
        max_duration: Optional[float]=None,
        num_demo_use: Optional[int]=-1,
        stereo_method: str="sensor",       # copy or sensor
    ):
        self.embodiment = embodiment
        self.num_demo_use = num_demo_use
        self.robot_action_dim = robot_action_dim
        self.gripper_action_dim = gripper_action_dim
        episode_ends = replay_buffer.episode_ends[:]
        self.repeat_frame_prob = repeat_frame_prob
        self.stereo_method = stereo_method
        assert repeat_frame_prob == 0.0, "repeat_frame_prob is not supported yet"

        # create indices, including (current_idx, start_idx, end_idx)
        indices = list()
        n_use_demo = np.min([len(episode_ends), num_demo_use]) if num_demo_use > 0 else len(episode_ends)
        if num_demo_use < len(episode_ends):
            print(f"Using {n_use_demo} demos out of {len(episode_ends)}")
        
        sequence_length = (key_horizon['action'] - 1) * key_down_sample_steps['action']
        sequence_pad_valid_length = int(sequence_length * (1.0 - action_padding_ratio))

        for i in range(n_use_demo):
            if episode_mask is not None and not episode_mask[i]:
                # skip episode
                continue
            start_idx = 0 if i == 0 else episode_ends[i-1]
            end_idx = episode_ends[i]
            if max_duration is not None:
                end_idx = min(end_idx, max_duration * 60)

            for current_idx in range(start_idx, end_idx):
                # TODO: this part ?? 
                if end_idx < current_idx + sequence_pad_valid_length + 1:
                    continue
                indices.append((current_idx, start_idx, end_idx))
                # current_idx, start_idx of the episode, end_idx of the episode, default_False
        
        # load low_dim to memory and keep rgb as compressed zarr array
        self.replay_buffer = dict()
        self.num_robot = 0
        for key in lowdim_keys:
            if key == 'task_id':
                # for task_id, we generate it from the HRASequenceSamplerWrapper
                # diffusion_policy/dataset/hra_dataset.py
                continue
            if key.endswith('eef_pos'):
                self.num_robot += 1
            if key.endswith('pos_abs'):
                axis = shape_meta['obs'][key]['axis']
                if isinstance(axis, int):
                    axis = [axis]
                self.replay_buffer[key] = replay_buffer[key[:-4]][:, list(axis)]
            elif key.endswith('quat_abs'):
                axis = shape_meta['obs'][key]['axis']
                if isinstance(axis, int):
                    axis = [axis]
                # HACK for hybrid abs/relative proprioception
                rot_in = replay_buffer[key[:-4]][:]
                rot_out = st.Rotation.from_quat(rot_in).as_euler('XYZ')
                self.replay_buffer[key] = rot_out[:, list(axis)]
            elif key.endswith('axis_angle_abs'):
                axis = shape_meta['obs'][key]['axis']
                if isinstance(axis, int):
                    axis = [axis]
                rot_in = replay_buffer[key[:-4]][:]
                rot_out = st.Rotation.from_rotvec(rot_in).as_euler('XYZ')
                self.replay_buffer[key] = rot_out[:, list(axis)]
            elif key.endswith('gripper_force'):
                if key in replay_buffer.keys():
                    self.replay_buffer[key] = replay_buffer[key][:]
                else:
                    # For human data, there is no gripper force, therefore we need to create a dummy gripper force
                    force_shape = shape_meta['obs'][key]['shape'][0]
                    self.replay_buffer[key] = np.zeros((replay_buffer['action'].shape[0], force_shape), dtype=np.float32)
                    print(f"Warning: gripper_force key [{key}] not found in replay buffer, using zeros{(replay_buffer['action'].shape[0], force_shape)} instead.")
            else:
                self.replay_buffer[key] = replay_buffer[key][:]
        for key in rgb_keys:
            self.replay_buffer[key] = replay_buffer[key]
        if 'instruction' in replay_buffer.keys():
            self.replay_buffer['instruction'] = replay_buffer['instruction']
        
        
        if 'action' in replay_buffer:
            self.replay_buffer['action'] = replay_buffer['action'][:]
        else:
            raise NotImplementedError("action key not found in replay buffer")
        
        if 'camera0_pose' in replay_buffer:
            self.camera_pose_arr = replay_buffer['camera0_pose'][:]
        else:
            if self.embodiment == 'human':
                raise NotImplementedError("camera0_pose key not found in replay buffer for human data")
            print("Warning: camera0_pose key not found in replay buffer, using zeros instead.")
            self.camera_pose_arr = np.zeros((replay_buffer['action'].shape[0], 6), dtype=np.float32)

        self.action_padding_ratio = action_padding_ratio
        self.indices = indices
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.key_horizon = key_horizon
        self.key_latency_steps = key_latency_steps
        self.key_down_sample_steps = key_down_sample_steps
        
        self.ignore_rgb_is_applied = False # speed up the interation when getting normalizaer

    def __len__(self):
        return len(self.indices)
    
    def sample_instruction(self, idx):
        current_idx, start_idx, end_idx = self.indices[idx]
        if 'instruction' in self.replay_buffer:
            return self.replay_buffer['instruction'][current_idx]    
        else:
            return None      

    def sample_sequence(self, idx):
            
        current_idx, start_idx, end_idx = self.indices[idx]

        result = dict()

        obs_keys = self.rgb_keys + self.lowdim_keys
        if self.ignore_rgb_is_applied:
            obs_keys = self.lowdim_keys

        cam_proj = np.linalg.inv(pose_to_mat(self.camera_pose_arr[current_idx])) @ pose_to_mat(self.camera_pose_arr[start_idx])    
        # observation
        for key in obs_keys:
            if key == 'task_id':
                # for task_id, we generate it from the HRASequenceSamplerWrapper
                # diffusion_policy/dataset/hra_dataset.py
                continue
            input_arr = self.replay_buffer[key]
            pair_input_arr = None
            this_horizon = self.key_horizon[key]
            this_latency_steps = self.key_latency_steps[key]
            this_downsample_steps = self.key_down_sample_steps[key]
            if 'eef_pos' in key:
                pair_key = key.replace('eef_pos', 'eef_rot_axis_angle')
                pair_this_horizon = self.key_horizon[pair_key]
                pair_this_latency_steps = self.key_latency_steps[pair_key]
                pair_this_downsample_steps = self.key_down_sample_steps[pair_key]
                assert this_horizon == pair_this_horizon
                assert this_latency_steps == pair_this_latency_steps
                assert this_downsample_steps == pair_this_downsample_steps
                pair_input_arr = self.replay_buffer[pair_key]
            elif 'eef_rot_axis_angle' in key:
                # we handle the robot eef rotation with eef_pos key
                continue
            elif 'gripper_pose' in key or 'gripper_force' in key or 'rgb' in key or 'embodiment' in key or 'pointcloud' in key:
                pass
            else:
                raise NotImplementedError('key {} not supported, currently only support eef_pos and eef_rot_axis_angle'.format(key))
            
            if key in self.rgb_keys:
                assert this_latency_steps == 0
                if type(this_downsample_steps) == int:
                    num_valid = min(this_horizon, (current_idx - start_idx) // this_downsample_steps + 1)
                    slice_start = current_idx - (num_valid - 1) * this_downsample_steps

                    output = input_arr[slice_start: current_idx + 1: this_downsample_steps]
                    assert output.shape[0] == num_valid
                    
                    # solve padding
                    if output.shape[0] < this_horizon:
                        padding = np.repeat(output[:1], this_horizon - output.shape[0], axis=0)
                        output = np.concatenate([padding, output], axis=0)
                elif type(this_downsample_steps) == omegaconf.listconfig.ListConfig or type(this_downsample_steps) == list:
                    target_idx = np.array([current_idx] + 
                        [current_idx - this_downsample_steps[idx] for idx in range(this_horizon - 1)])
                    target_idx = target_idx[target_idx >= start_idx][::-1]
                    output = input_arr[target_idx]
                    # solve padding
                    if output.shape[0] < this_horizon:
                        padding = np.repeat(output[:1], this_horizon - output.shape[0], axis=0)
                        output = np.concatenate([padding, output], axis=0)
                result[key] = output
            else:
                if type(this_downsample_steps) == int:
                    idx_with_latency = np.array(
                        [current_idx - idx * this_downsample_steps + this_latency_steps for idx in range(this_horizon)],
                        dtype=np.float32)
                elif type(this_downsample_steps) == omegaconf.listconfig.ListConfig:
                    idx_with_latency = np.array([current_idx] + 
                        [current_idx - this_downsample_steps[idx] + this_latency_steps for idx in range(this_horizon - 1)],
                        dtype=np.float32)
                idx_with_latency = idx_with_latency[::-1]  # small_idx -> large_idx
                idx_with_latency = np.clip(idx_with_latency, start_idx, end_idx - 1)
                interpolation_start = max(int(idx_with_latency[0]) - 5, start_idx)
                interpolation_end = min(int(idx_with_latency[-1]) + 2 + 5, end_idx)

                if 'eef_pos' in key:   # pair key is eef_rot_axis_angle
                    # rotation
                    rot_preprocess, rot_postprocess = None, None
                    if pair_key.endswith('quat'):
                        rot_preprocess = st.Rotation.from_quat
                        rot_postprocess = st.Rotation.as_quat
                    elif pair_key.endswith('axis_angle'):
                        rot_preprocess = st.Rotation.from_rotvec
                        rot_postprocess = st.Rotation.as_rotvec
                    else:
                        raise NotImplementedError
                    slerp = st.Slerp(
                        times=np.arange(interpolation_start, interpolation_end),
                        rotations=rot_preprocess(pair_input_arr[interpolation_start: interpolation_end]))
                    output_rot = rot_postprocess(slerp(idx_with_latency))
                    interp = si.interp1d(
                        x=np.arange(interpolation_start, interpolation_end),
                        y=input_arr[interpolation_start: interpolation_end],
                        axis=0, assume_sorted=True)
                    output_pos = interp(idx_with_latency)

                    # need back projection since camera pose may change
                    output = np.concatenate([output_pos, output_rot], axis=-1)
                    output = pose_to_mat(output)
                    output = cam_proj @ output
                    output = mat_to_pose(output)
                    result[key] = output[:, :3]
                    result[pair_key] = output[:, 3:]
                elif 'gripper_pose' in key or 'gripper_force' in key or 'embodiment' in key:
                    interp = si.interp1d(
                        x=np.arange(interpolation_start, interpolation_end),
                        y=input_arr[interpolation_start: interpolation_end],
                        axis=0, assume_sorted=True)
                    result[key] = interp(idx_with_latency)
                else:
                    raise NotImplementedError('key {} not supported, currently only support eef_pos and eef_rot_axis_angle'.format(key))

        # aciton
        input_arr = self.replay_buffer['action']
        action_horizon = self.key_horizon['action']
        action_latency_steps = self.key_latency_steps['action']
        assert action_latency_steps == 0
        action_down_sample_steps = self.key_down_sample_steps['action']
        slice_end = min(end_idx, current_idx + (action_horizon - 1) * action_down_sample_steps + 1)
        output = input_arr[current_idx: slice_end: action_down_sample_steps].copy()
        
        st_action_dim = 0
        for idx, robot_dim in enumerate(self.robot_action_dim):
            assert robot_dim == 6 
            cam0_action = output[:, st_action_dim:st_action_dim + 6]
            cam0_action = pose_to_mat(cam0_action)
            cam_action = cam_proj @ cam0_action
            cam_action = mat_to_pose(cam_action)
            output[:, st_action_dim:st_action_dim + 6] = cam_action
            st_action_dim = st_action_dim + robot_dim + self.gripper_action_dim[idx]
        
        # solve padding
        if self.action_padding_ratio == 0.0:
            assert output.shape[0] == action_horizon
        elif output.shape[0] < action_horizon:
            padding = np.repeat(output[-1:], action_horizon - output.shape[0], axis=0)
            output = np.concatenate([output, padding], axis=0)
        result['action'] = output

        for key in self.rgb_keys:
            if key not in result.keys():
                continue
            if '_right' in key and self.stereo_method == 'copy':
                copy_key = key.replace('_right', '')
                result[key] = result[copy_key]
        
        if 'instruction' in self.replay_buffer:
            result['instruction'] = self.replay_buffer['instruction'][current_idx]        
        return result
    
    def ignore_rgb(self, apply=True):
        self.ignore_rgb_is_applied = apply