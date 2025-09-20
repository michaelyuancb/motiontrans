import copy
from typing import Dict, Optional

import os
from datetime import datetime
import pathlib
import numpy as np
import torch
import zarr
import random
from PIL import Image
from pathlib import Path
from threadpoolctl import threadpool_limits
from tqdm import trange, tqdm
from filelock import FileLock
import shutil

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from common.normalize_util import (
    array_to_stats, concatenate_normalizer, get_identity_normalizer_from_stat,
    get_image_identity_normalizer, get_range_normalizer_from_stat)
from common.pose_repr_util import convert_pose_mat_rep
from common.pytorch_util import dict_apply
from common.replay_buffer import ReplayBuffer
from common.hra_sampler import HRASequenceSampler, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from common.pose_util import pose_to_mat, mat_to_pose10d
from huggingface_hub import hf_hub_download, snapshot_download
from diffusion_policy.dataset.hra_dataset import get_replay_buffer_list, HRASequenceSamplerWrapper


class HRA3Dataset(BaseDataset):  # human-robot-action dataset
    def __init__(self,
        shape_meta: dict,
        dataset_path: str,
        human_dataset_path: Optional[str]=None,
        alpha: Optional[float]=1.0,          # alpha for real-robot data, refer to https://arxiv.org/abs/2503.22634
        cache_dir: Optional[str]=None,
        pose_repr: dict={},
        action_padding: bool=False,
        temporally_independent_normalization: bool=False,
        repeat_frame_prob: float=0.0,
        seed: int=42,
        val_ratio: float=0.0,
        max_duration: Optional[float]=None,
        use_ratio: float=1.0,
        dataset_idx: Optional[str]=None,
        num_demo_use: Optional[int]=-1, 
    ):
        if alpha != 1.0:
            assert human_dataset_path is not None, 'alpha is not 1.0, please provide human dataset path.'

        if num_demo_use > 0:
            # only for finetuning
            assert human_dataset_path is None, 'num_demo_use is set to be > 0, only for robot finetuning, please do not provide human dataset path.'

        self.pose_repr = pose_repr
        self.obs_pose_repr = self.pose_repr.get('obs_pose_repr', 'relative')
        self.action_pose_repr = self.pose_repr.get('action_pose_repr', 'relative')
        self.robot_action_dim = shape_meta['action']['robot_action_dim']
        self.gripper_action_dim = shape_meta['action']['gripper_action_dim']
        self.num_robot = len(self.robot_action_dim)
        self.num_gripper = len(self.gripper_action_dim)
        
        replay_buffer_list, real_sampler_idx = get_replay_buffer_list(dataset_path, cache_dir)
        if (human_dataset_path is not None) and (alpha < 1.0):
            human_replay_buffer_list, real_sampler_idx_human = get_replay_buffer_list(human_dataset_path, cache_dir)
            real_sampler_idx_human = [i + max(real_sampler_idx) + 1 for i in real_sampler_idx_human]
            real_sampler_idx = real_sampler_idx + real_sampler_idx_human
        else:
            human_replay_buffer_list = None

        rgb_keys = list()
        lowdim_keys = list()
        key_horizon = dict()
        key_down_sample_steps = dict()
        key_latency_steps = dict()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            # solve obs type
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)

            # solve obs_horizon
            horizon = shape_meta['obs'][key]['horizon']
            key_horizon[key] = horizon

            # solve latency_steps
            latency_steps = shape_meta['obs'][key]['latency_steps']
            key_latency_steps[key] = latency_steps

            # solve down_sample_steps
            down_sample_steps = shape_meta['obs'][key]['down_sample_steps']
            key_down_sample_steps[key] = down_sample_steps

        # solve action
        key_horizon['action'] = shape_meta['action']['horizon']
        key_latency_steps['action'] = shape_meta['action']['latency_steps']
        key_down_sample_steps['action'] = shape_meta['action']['down_sample_steps']

        n_human_episodes = 0 
        if human_replay_buffer_list is None:
            pass  
        else:
            for human_replay_buffer in human_replay_buffer_list:
                n_human_episodes = n_human_episodes + human_replay_buffer.n_episodes
        n_robot_episodes = 0
        for replay_buffer in replay_buffer_list:
            n_robot_episodes = n_robot_episodes + replay_buffer.n_episodes
        if dataset_idx is None:
            if val_ratio == 0.0:
                val_mask = np.zeros(n_robot_episodes + n_human_episodes, dtype=bool)
            else:
                val_mask = get_val_mask(
                    n_episodes=n_robot_episodes + n_human_episodes, 
                    val_ratio=val_ratio,
                    seed=seed
                )
        
            train_mask = ~val_mask
        assert use_ratio <= 1.0

        print('use total episode number:', train_mask.shape[0])
        print('use training episode number:', (train_mask == True).sum())
        print('use training robot episode number:', (train_mask[:replay_buffer.n_episodes] == True).sum())
        print('use training human episode number:', (train_mask[replay_buffer.n_episodes:] == True).sum())

        self.sampler_lowdim_keys = list()
        for key in lowdim_keys:
            self.sampler_lowdim_keys.append(key)

        sampler = HRASequenceSamplerWrapper(
            alpha=alpha,
            shape_meta=shape_meta,
            replay_buffer_list=replay_buffer_list,
            human_replay_buffer_list=human_replay_buffer_list,
            real_sampler_idx=real_sampler_idx,
            rgb_keys=rgb_keys,
            lowdim_keys=self.sampler_lowdim_keys,
            key_horizon=key_horizon,
            key_latency_steps=key_latency_steps,
            key_down_sample_steps=key_down_sample_steps,
            robot_action_dim=self.robot_action_dim,
            gripper_action_dim=self.gripper_action_dim,
            episode_mask=train_mask,
            action_padding=action_padding,
            repeat_frame_prob=0.0,
            max_duration=max_duration,
            num_demo_use=num_demo_use,
        )
        
        self.train_mask = train_mask
        self.n_robot_episodes = replay_buffer.n_episodes
        self.n_human_episodes = n_human_episodes
        self.alpha = alpha
        self.shape_meta = shape_meta
        self.replay_buffer = replay_buffer
        self.human_replay_buffer = human_replay_buffer
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.key_horizon = key_horizon
        self.key_latency_steps = key_latency_steps
        self.key_down_sample_steps = key_down_sample_steps
        self.val_mask = val_mask
        self.action_padding = action_padding
        self.repeat_frame_prob = repeat_frame_prob
        self.max_duration = max_duration
        self.num_demo_use = num_demo_use
        self.sampler = sampler
        self.temporally_independent_normalization = temporally_independent_normalization
        self.threadpool_limits_is_applied = False

        # We use MIL-Texture to achieve fast and batch-upd random background augmentation
        # Details:
        # https://roboengine.github.io/  &&  https://greenaug.github.io/
        background_root = snapshot_download(repo_id="eugeneteoh/mil_data", repo_type="dataset", allow_patterns="*.png")
        self.texture_pool = sorted(Path(background_root).glob("**/*.png"))


    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = HRASequenceSamplerWrapper(
            alpha=self.alpha,
            shape_meta=self.shape_meta,
            replay_buffer_list=self.replay_buffer_list,
            human_replay_buffer_list=self.human_replay_buffer_list,
            real_sampler_idx=self.sampler.real_sampler_idx,
            rgb_keys=self.rgb_keys,
            lowdim_keys=self.sampler_lowdim_keys,
            key_horizon=self.key_horizon,
            key_latency_steps=self.key_latency_steps,
            key_down_sample_steps=self.key_down_sample_steps,
            robot_action_dim=self.robot_action_dim,
            gripper_action_dim=self.gripper_action_dim,
            episode_mask=self.val_mask,
            action_padding=self.action_padding,
            repeat_frame_prob=self.repeat_frame_prob,
            max_duration=self.max_duration,
            num_demo_use=self.num_demo_use,
        )
        val_set.val_mask = ~self.val_mask
        return val_set
    
    def _get_normalizer(self, data_cache):
        normalizer = LinearNormalizer()

        if len(data_cache) == 0:
            return normalizer

        for key in data_cache.keys():
            data_cache[key] = np.concatenate(data_cache[key])
            # assert data_cache[key].shape[0] == len(self.sampler)
            assert len(data_cache[key].shape) == 3
            B, T, D = data_cache[key].shape
            if not self.temporally_independent_normalization:
                data_cache[key] = data_cache[key].reshape(B*T, D)

        # action
        action_normalizers = list()
        st_action_dim = 0 
        for idx, robot_dim in enumerate(self.robot_action_dim):
            assert robot_dim == 6
            action_normalizers.append(get_range_normalizer_from_stat(array_to_stats(data_cache['action'][..., st_action_dim: st_action_dim + 3])))  # pos
            action_normalizers.append(get_identity_normalizer_from_stat(array_to_stats(data_cache['action'][..., st_action_dim + 3: st_action_dim + 9]))) # rot, attention that it is 6d-rep from batch ! 
            st_action_dim = st_action_dim + 9             # watch out ! we use 9 dof action representation
            gripper_dim = self.gripper_action_dim[idx]
            action_normalizers.append(get_range_normalizer_from_stat(array_to_stats(data_cache['action'][..., st_action_dim: st_action_dim + gripper_dim])))  # gripper
            st_action_dim = st_action_dim + gripper_dim
        normalizer['action'] = concatenate_normalizer(action_normalizers)

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(data_cache[key])

            if key.endswith('pos') or 'pos_wrt' in key:
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('pos_abs'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('rot_axis_angle') or 'rot_axis_angle_wrt' in key:
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('gripper_width'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('gripper_pose'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('gripper_force'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('task_id'):
                this_normalizer = get_identity_normalizer_from_stat(stat)
            else:
                raise RuntimeError('unsupported')
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_identity_normalizer()
        
        return normalizer

    
    def get_normalizer(self, use_embodiment_normalizer=False, **kwargs) -> LinearNormalizer:

        # enumerate the dataset and save low_dim data
        self.sampler.ignore_rgb(True)
        batch_size = 1 if use_embodiment_normalizer else 32
        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            num_workers=32,
        )
        data_cache = {key: list() for key in self.lowdim_keys + ['action']}
        if use_embodiment_normalizer:
            data_cache_human = {key: list() for key in self.lowdim_keys + ['action']}
        for batch in tqdm(dataloader, desc='iterating dataset to get normalization'):
            if not use_embodiment_normalizer:
                for key in self.lowdim_keys:
                    data_cache[key].append(copy.deepcopy(batch['obs'][key]))
                data_cache['action'].append(copy.deepcopy(batch['action']))
            else:
                n_data = batch['embodiment'].shape[0]
                for i in range(n_data):
                    if batch['embodiment'][i].item() == 1:
                        for key in self.lowdim_keys:
                            data_cache[key].append(copy.deepcopy(batch['obs'][key][i:i+1]))
                        data_cache['action'].append(copy.deepcopy(batch['action'][i:i+1]))
                    else:
                        for key in self.lowdim_keys:
                            data_cache_human[key].append(copy.deepcopy(batch['obs'][key][i:i+1]))
                        data_cache_human['action'].append(copy.deepcopy(batch['action'][i:i+1]))
        self.sampler.ignore_rgb(False)
        if use_embodiment_normalizer:
            return self._get_normalizer(data_cache), self._get_normalizer(data_cache_human)
        else:
            return self._get_normalizer(data_cache), None

    def get_num_task(self):
        # for each subset, it represent one task
        return self.sampler.get_num_task()

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.threadpool_limits_is_applied:
            threadpool_limits(1)
            self.threadpool_limits_is_applied = True
        data = self.sampler.sample_sequence(idx)

        obs_dict = dict()
        for key in self.rgb_keys:
            if not key in data:
                continue
            if 'rgb' in key:
                # move channel last to channel first
                # T,H,W,C
                # convert uint8 image to float32
                if 'mask' in key:
                    continue 
                rgb_obs = data[key]            # (T, H, W, C)
                mask_obs = data[key+"_mask"]   # (T, H, W), where 1 is embodiment, 0 is background

                # Conducting Background Augmentation
                bg_path = random.choice(self.texture_pool)
                h, w = mask_obs.shape[1:]
                background = Image.open(bg_path).resize((w, h), Image.Resampling.LANCZOS)
                background = np.array(background)
                if background.ndim == 2:
                    background = background[..., None]
                background = background[None]            # (1, H, W, C)
                rgb_obs = mask_obs[..., None] * rgb_obs + (1 - mask_obs[..., None]) * background
                obs_dict[key] = np.moveaxis(rgb_obs, -1, 1).astype(np.float32) / 255.
                # Image.fromarray((obs_dict['camera0_rgb'][0].transpose(1,2,0) * 255).astype(np.uint8)).save("rgb.png")
                # T,C,H,W
            elif 'pointcloud' in key:
                # T,N,C
                obs_dict[key] = data[key].astype(np.float32)
            else:
                raise RuntimeError(f'Unsupported key {key} in rgb_keys, please check the dataset shape meta.')
            del data[key]
        for key in self.sampler_lowdim_keys:
            obs_dict[key] = data[key].astype(np.float32)
            del data[key]
        
        # generate relative pose between two ees
        for robot_id in range(self.num_robot):
            # convert pose to mat
            pose_mat = pose_to_mat(np.concatenate([
                obs_dict[f'robot{robot_id}_eef_pos'],
                obs_dict[f'robot{robot_id}_eef_rot_axis_angle']
            ], axis=-1))
            for other_robot_id in range(self.num_robot):
                if robot_id == other_robot_id:
                    continue
                if not f'robot{robot_id}_eef_pos_wrt{other_robot_id}' in self.lowdim_keys:
                    continue
                other_pose_mat = pose_to_mat(np.concatenate([
                    obs_dict[f'robot{other_robot_id}_eef_pos'],
                    obs_dict[f'robot{other_robot_id}_eef_rot_axis_angle']
                ], axis=-1))
                rel_obs_pose_mat = convert_pose_mat_rep(
                    pose_mat,
                    base_pose_mat=other_pose_mat[-1],
                    pose_rep=self.obs_pose_repr,
                    backward=False)
                rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
                obs_dict[f'robot{robot_id}_eef_pos_wrt{other_robot_id}'] = rel_obs_pose[:,:3]
                obs_dict[f'robot{robot_id}_eef_rot_axis_angle_wrt{other_robot_id}'] = rel_obs_pose[:,3:]

        del_keys = list()
        for key in obs_dict:
            if key.endswith('_demo_start_pose') or key.endswith('_demo_end_pose'):
                del_keys.append(key)
        for key in del_keys:
            del obs_dict[key]

        actions = list()
        for robot_id in range(self.num_robot):
            # convert pose to mat
            pose_mat = pose_to_mat(np.concatenate([
                obs_dict[f'robot{robot_id}_eef_pos'],
                obs_dict[f'robot{robot_id}_eef_rot_axis_angle']
            ], axis=-1))
            action_mat = pose_to_mat(data['action'][...,6 * robot_id: 6 * robot_id + 6])
            
            # solve relative obs
            obs_pose_mat = convert_pose_mat_rep(
                pose_mat, 
                base_pose_mat=pose_mat[-1],
                pose_rep=self.obs_pose_repr,
                backward=False)
            action_pose_mat = convert_pose_mat_rep(
                action_mat, 
                base_pose_mat=pose_mat[-1],
                pose_rep=self.action_pose_repr,
                backward=False)
        
            # convert pose to pos + rot6d representation
            obs_pose = mat_to_pose10d(obs_pose_mat)
            action_pose = mat_to_pose10d(action_pose_mat)
            actions.append(action_pose)

            # generate data
            obs_dict[f'robot{robot_id}_eef_pos'] = obs_pose[:,:3]
            obs_dict[f'robot{robot_id}_eef_rot_axis_angle'] = obs_pose[:,3:]
        
        actions.append(data['action'][..., 6*self.num_robot:])
        data['action'] = np.concatenate(actions, axis=-1)
        
        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(data['action'].astype(np.float32)),
            'alpha': torch.tensor(data['alpha'], dtype=torch.float32),
            'embodiment': torch.tensor(data['embodiment'], dtype=torch.float32),
        }
        return torch_data
