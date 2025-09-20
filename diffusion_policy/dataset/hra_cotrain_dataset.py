import copy
import random
import numpy as np
from typing import Dict, Optional
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from common.hra_sampler import get_val_mask
from diffusion_policy.dataset.hra_dataset import HRASequenceSamplerWrapper, HRADataset, get_replay_buffer_list

class HRACoTrainSequenceSamplerWrapper(HRASequenceSamplerWrapper):

    def sample_sequence(self, idx: int) -> dict:

        idx_org = idx
        if idx < self.n_robot_sampler_idx:
            pass 
        else:
            # for cotraining mode, we random select a help-datapoint from co-training data
            idx = random.randint(self.n_robot_sampler_idx, self.sampler_idx_np[-1] - 1)

        sampler_idx = int(np.where(idx >= self.sampler_idx_np)[0].max())
        _ = self.sampler_list[sampler_idx]
        data = self.sampler_list[sampler_idx].sample_sequence(idx - self.sampler_idx_np[sampler_idx])
        data['task_id'] = np.ones((1, 1)) * self.real_sampler_idx[sampler_idx]
        if idx < self.n_robot_sampler_idx:
            data['alpha'] = self.adjust_alpha
            data['embodiment'] = 1    # robot
        else:
            data['alpha'] = 1.0 - self.adjust_alpha
            data['embodiment'] = 0    # human
        return data

    def __len__(self): 
        return 2 * self.n_robot_sampler_idx


class HRACoTrainDataset(HRADataset):  # human-robot-action dataset

    """
    For CoTraining, we assume dataset_path as the task-target robot dataset, human_dataset_path as the co-training helper dataset,
    but note that this 'human_dataset_path' could involve both robot and human data. 
    """

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

        assert val_ratio == 0.0, f"HRACoTrain Dataset does not support val_ratio > 0.0. Your val_ratio: {val_ratio}"
    
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
        print('use training robot episode number:', (train_mask[:n_robot_episodes] == True).sum())
        print('use training human episode number:', (train_mask[n_robot_episodes:] == True).sum())

        self.sampler_lowdim_keys = list()
        for key in lowdim_keys:
            self.sampler_lowdim_keys.append(key)

        sampler = HRACoTrainSequenceSamplerWrapper(
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
        self.n_robot_episodes = n_robot_episodes
        self.n_human_episodes = n_human_episodes
        self.alpha = alpha
        self.shape_meta = shape_meta
        self.replay_buffer_list = replay_buffer_list
        self.human_replay_buffer_list = human_replay_buffer_list
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

    
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = HRACoTrainSequenceSamplerWrapper(
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