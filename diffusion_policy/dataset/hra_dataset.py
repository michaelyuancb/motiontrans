import copy
from typing import Dict, Optional

import os
from datetime import datetime
import pathlib
import numpy as np
import torch
import zarr
import hashlib
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

register_codecs()

def get_replay_buffer(dataset_path, cache_dir):
    if dataset_path is None:
        return None
    if cache_dir is None:
        replay_buffer = ReplayBuffer.create_from_path(zarr_path=dataset_path, mode='r')      
    else:
        # TODO: refactor into a stand alone function?
        # determine path name
        mod_time = os.path.getmtime(dataset_path)
        stamp = datetime.fromtimestamp(mod_time).isoformat()
        stem_name = os.path.basename(dataset_path).split('.')[0]
        cache_name = '_'.join([stem_name, stamp])
        cache_dir = pathlib.Path(os.path.expanduser(cache_dir))
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir.joinpath(cache_name + '.zarr.mdb')
        lock_path = cache_dir.joinpath(cache_name + '.lock')
        
        # load cached file
        print('Acquiring lock on cache.')
        with FileLock(lock_path):
            # cache does not exist
            if not cache_path.exists():
                try:
                    with zarr.LMDBStore(str(cache_path),     
                        writemap=True, metasync=False, sync=False, map_async=True, lock=False
                    ) as lmdb_store:
                        print(f"Copying data to {str(cache_path)}")
                        ReplayBuffer.copy_from_path(zarr_path=dataset_path, store=lmdb_store, compressors='disk')
                    print("Cache written to disk!")
                except Exception as e:
                    shutil.rmtree(cache_path)
                    raise e
            
        # open read-only lmdb store
        store = zarr.LMDBStore(str(cache_path), readonly=True, lock=False)
        replay_buffer = ReplayBuffer.create_from_group(
            group=zarr.group(store)
        )
    return replay_buffer


def get_real_sampler_idx(dataset_path_list):
    real_name_list = []
    for data_p in dataset_path_list:
        if data_p.endswith('_aug.zarr'):
            real_name = data_p[:-len('_aug.zarr')]
        elif data_p.endswith('_masks.zarr'):
            real_name = data_p[:-len('_masks.zarr')]
        elif data_p.endswith('.zarr'):
            real_name = data_p[:-len('.zarr')]
        else:
            raise ValueError(f'Unsupported dataset path {data_p} (only support _aug, _masks and original .zarr), please check the dataset path.')
        if real_name not in real_name_list:
            real_name_list.append(real_name)

    real_name_list_dict = dict()
    for idx, real_name in enumerate(real_name_list):
        real_name_list_dict[real_name] = idx
    
    real_sampler_idx = []
    for data_p in dataset_path_list:
        if data_p.endswith('_aug.zarr'):
            real_name = data_p[:-len('_aug.zarr')]
        elif data_p.endswith('_masks.zarr'):
            real_name = data_p[:-len('_masks.zarr')]
        elif data_p.endswith('.zarr'):
            real_name = data_p[:-len('.zarr')]
        else:
            raise ValueError(f'Unsupported dataset path {data_p} (only support _aug, _masks and original .zarr), please check the dataset path.')
        real_sampler_idx.append(real_name_list_dict[real_name])

    return real_name_list, real_sampler_idx


def get_replay_buffer_list(dataset_path, cache_dir):
    dataset_path_list_tmp = dataset_path.split("|")
    dataset_path_list_tmp = [data_p for data_p in dataset_path_list_tmp if data_p is not None and len(data_p) > 0]
    dataset_path_list = []
    for data_p in dataset_path_list_tmp:
        if data_p.endswith(".json"):
            continue  # skip json files
        if data_p.endswith('.zarr'):
            dataset_path_list.append(data_p)
        else:
            data_p_list = os.listdir(data_p)
            for data_fp in data_p_list:
                if data_fp.endswith('.zarr'):
                    dataset_path_list.append(os.path.join(data_p, data_fp))
                elif data_fp.endswith(".json"):
                    continue  # skip json files
                else:
                    raise ValueError(f'Unsupported dataset path {data_fp} from auto-folder file finding, only support .zarr files, please check the dataset path.')
    replay_buffer_list = []
    for data_p in dataset_path_list:
        print(data_p)
        replay_buffer_list.append(get_replay_buffer(data_p, cache_dir))
    real_name_list, real_sampler_idx = get_real_sampler_idx(dataset_path_list)
    return replay_buffer_list, dataset_path_list, real_sampler_idx, real_name_list


def get_instruction_from_filename_list(filename_list):
    instruction_list = []
    for filename in filename_list:
        # instruction位于两个+中间，例如XXXX+instruction+XXXX
        if '+' in filename and filename.find('+') > 0:
            instruction = filename[filename.find('+') + 1:filename.rfind('+')]
            instruction = instruction.strip()
            instruction = instruction.replace('_', ' ')
            if not instruction.endswith('.'):
                instruction += '.'
            if '+' in instruction:
                raise ValueError(f'Filename {filename} contains multiple instructions between + signs.')
            if len(instruction) > 0:
                instruction_list.append(instruction)
            else:
                raise ValueError(f'Filename {filename} contains empty instruction between + signs.')
        else:
            raise ValueError(f'Filename {filename} does not contain instruction in +task_name+ format.')
    return instruction_list


class HRASequenceSamplerWrapper:

    def __init__(self,
        alpha,
        shape_meta,
        replay_buffer_list,
        human_replay_buffer_list,
        real_sampler_idx,
        real_instruction_list, 
        rgb_keys: list,
        lowdim_keys: list,
        key_horizon: dict,
        key_latency_steps: dict,
        key_down_sample_steps: dict,
        robot_action_dim: list,
        gripper_action_dim: list,
        episode_mask: np.ndarray,
        action_padding_ratio: float,
        repeat_frame_prob: float,
        max_duration: float,
        num_demo_use: int,
        human_stereo_method: str
    ):
        
        self.real_sampler_idx = real_sampler_idx
        self.real_instruction_list = real_instruction_list
        self.robot_episode_bin = []
        self.human_episode_bin = []
        for replay_buffer in replay_buffer_list:
            self.robot_episode_bin.append(replay_buffer.n_episodes)
        if human_replay_buffer_list is not None:
            for human_replay_buffer in human_replay_buffer_list:
                self.human_episode_bin.append(human_replay_buffer.n_episodes)

        self.n_robot_episodes = sum(self.robot_episode_bin) if len(self.robot_episode_bin) > 0 else 0
        self.n_human_episodes = sum(self.human_episode_bin) if len(self.human_episode_bin) > 0 else 0
        self.alpha = alpha
        if self.n_human_episodes == 0.0:
            self.adjust_alpha = 1.0 
        else:
            # NOTE: we adjust the alpha for human-robot mixing, fix the mis-definition of https://arxiv.org/abs/2503.22634
            alpha_robot = alpha / self.n_robot_episodes
            alpha_human = (1 - alpha) / self.n_human_episodes
            self.adjust_alpha = alpha_robot / (alpha_robot + alpha_human)
        st_episode_mask = 0
        self.sampler_list = []
        for replay_buffer in replay_buffer_list:
            n_episode = replay_buffer.n_episodes
            sampler = HRASequenceSampler(
                embodiment="robot",
                shape_meta=shape_meta,
                replay_buffer=replay_buffer,
                rgb_keys=rgb_keys,
                lowdim_keys=lowdim_keys,
                key_horizon=key_horizon,
                key_latency_steps=key_latency_steps,
                key_down_sample_steps=key_down_sample_steps,
                robot_action_dim=robot_action_dim,
                gripper_action_dim=gripper_action_dim,
                episode_mask=episode_mask[st_episode_mask:st_episode_mask + n_episode],
                action_padding_ratio=action_padding_ratio,
                repeat_frame_prob=repeat_frame_prob,
                max_duration=max_duration,
                num_demo_use=num_demo_use,
                stereo_method='sensor',
            )
            self.sampler_list.append(sampler)
            st_episode_mask = st_episode_mask + n_episode

        self.n_robot_sampler_idx = sum([len(sampler) for sampler in self.sampler_list])
        if human_replay_buffer_list is not None:
            for human_replay_buffer in human_replay_buffer_list:
                n_episode = human_replay_buffer.n_episodes
                human_sampler = HRASequenceSampler(
                    embodiment="human",
                    shape_meta=shape_meta,
                    replay_buffer=human_replay_buffer,
                    rgb_keys=rgb_keys,
                    lowdim_keys=lowdim_keys,
                    key_horizon=key_horizon,
                    key_latency_steps=key_latency_steps,
                    key_down_sample_steps=key_down_sample_steps,
                    robot_action_dim=robot_action_dim,
                    gripper_action_dim=gripper_action_dim,
                    episode_mask=episode_mask[st_episode_mask:st_episode_mask + n_episode],
                    action_padding_ratio=action_padding_ratio,
                    repeat_frame_prob=repeat_frame_prob,
                    max_duration=max_duration,
                    num_demo_use=num_demo_use,
                    stereo_method=human_stereo_method,
                )
                self.sampler_list.append(human_sampler)
                st_episode_mask = st_episode_mask + n_episode
        else:
            pass
        self.sampler_idx_list = []
        for sampler in self.sampler_list:
            self.sampler_idx_list.append(len(sampler))
        self.sampler_idx_np = np.array([0] + self.sampler_idx_list)
        self.sampler_idx_np = self.sampler_idx_np.cumsum(axis=0)

    def get_num_task(self):
        # for each subset, it represent one task
        return len(self.sampler_list)

    def ignore_rgb(self, ignore_rgb: bool):
        for sampler in self.sampler_list:
            sampler.ignore_rgb(ignore_rgb)

    def sample_instruction(self, idx: int) -> str:
        sampler_idx = int(np.where(idx >= self.sampler_idx_np)[0].max())
        instruction = self.sampler_list[sampler_idx].sample_instruction(idx - self.sampler_idx_np[sampler_idx])
        if instruction is None:
            instruction = self.real_instruction_list[sampler_idx].strip()
        return instruction

    def sample_sequence(self, idx: int) -> dict:
        sampler_idx = int(np.where(idx >= self.sampler_idx_np)[0].max())
        data = self.sampler_list[sampler_idx].sample_sequence(idx - self.sampler_idx_np[sampler_idx])
        data['task_id'] = np.ones((1, 1)) * self.real_sampler_idx[sampler_idx]
        # data['task_id'] = np.zeros((1, 1))
        if idx < self.n_robot_sampler_idx:
            data['alpha'] = self.adjust_alpha
            data['embodiment'] = 1    # robot
        else:
            data['alpha'] = 1.0 - self.adjust_alpha
            data['embodiment'] = 0    # human
        if 'instruction' not in data.keys():
            # For data without instruction, we add instruction from its filename.
            if self.real_instruction_list is not None and len(self.real_instruction_list) > 0:
                assert sampler_idx < len(self.real_instruction_list), \
                    f'sampler_idx {sampler_idx} is out of range of real_instruction_list {len(self.real_instruction_list)}'
                data['instruction'] = self.real_instruction_list[sampler_idx].strip()
        else:
            data['instruction'] = str(data['instruction']).strip()
        if self.real_instruction_list is not None:
            if not data['instruction'].endswith('.'):
                data['instruction'] += '.'
        return data

    def __len__(self):
        return self.sampler_idx_np[-1]


class HRADataset(BaseDataset):  # human-robot-action dataset
    def __init__(self,
        shape_meta: dict,
        dataset_path: str,
        human_dataset_path: Optional[str]=None,
        alpha: Optional[float]=1.0,          # alpha for real-robot data, refer to https://arxiv.org/abs/2503.22634
        cache_dir: Optional[str]=None,
        pose_repr: dict={},
        action_padding_ratio: float=1.0,
        temporally_independent_normalization: bool=False,
        repeat_frame_prob: float=0.0,
        seed: int=42,
        val_ratio: float=0.0,
        max_duration: Optional[float]=None,
        use_ratio: float=1.0,
        dataset_idx: Optional[str]=None,
        num_demo_use: Optional[int]=-1, 
        use_instruction: bool=False,
        text_feature_cache_dir: str="", 
        human_stereo_method: str="copy",   # copy or sensor
    ):
        assert human_stereo_method in ['copy', 'sensor']
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
        replay_buffer_list, dataset_path_list, real_sampler_idx, real_name_list = get_replay_buffer_list(dataset_path, cache_dir)
        print("Robot Dataset Path List:", dataset_path_list)
        if (human_dataset_path is not None) and (alpha < 1.0):
            human_replay_buffer_list, dataset_path_list_human, real_sampler_idx_human, real_name_list_human = get_replay_buffer_list(human_dataset_path, cache_dir)
            print("Human Dataset Path List:", dataset_path_list_human)
            real_sampler_idx_human = [i + max(real_sampler_idx) + 1 for i in real_sampler_idx_human]
            real_sampler_idx = real_sampler_idx + real_sampler_idx_human
            real_name_list = real_name_list + real_name_list_human
            dataset_path_list = dataset_path_list + dataset_path_list_human
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
            print(f"{key}: horizon={horizon}, latency_steps={latency_steps}, down_sample_steps={down_sample_steps}")
            key_down_sample_steps[key] = down_sample_steps

        self.use_instruction = use_instruction
        if use_instruction is False:
            assert text_feature_cache_dir is None or len(text_feature_cache_dir) == 0, \
                'use_instruction is False, please do not provide text_feature_cache_dir.'
        else:
            # assert text_feature_cache_dir is not None, \
            #     'use_instruction is True, please provide text_feature_cache_dir.'
            # assert len(text_feature_cache_dir) > 0, \
            #     'use_instruction is True, please provide text_feature_cache_dir.'
            if len(text_feature_cache_dir) == 0:
                print("text_feature_cache_dir is None")
        if len(text_feature_cache_dir) > 0:
            self.text_feature_cache_dir = text_feature_cache_dir
            os.makedirs(self.text_feature_cache_dir, exist_ok=True)
        else:
            self.text_feature_cache_dir = None
        if self.use_instruction:
            real_instruction_list = get_instruction_from_filename_list(real_name_list)
        else:
            real_instruction_list = None

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

        sampler = HRASequenceSamplerWrapper(
            alpha=alpha,
            shape_meta=shape_meta,
            replay_buffer_list=replay_buffer_list,
            human_replay_buffer_list=human_replay_buffer_list,
            real_sampler_idx=real_sampler_idx,
            real_instruction_list=real_instruction_list,
            rgb_keys=rgb_keys,
            lowdim_keys=self.sampler_lowdim_keys,
            key_horizon=key_horizon,
            key_latency_steps=key_latency_steps,
            key_down_sample_steps=key_down_sample_steps,
            robot_action_dim=self.robot_action_dim,
            gripper_action_dim=self.gripper_action_dim,
            episode_mask=train_mask,
            action_padding_ratio=action_padding_ratio,
            repeat_frame_prob=0.0,
            max_duration=max_duration,
            num_demo_use=num_demo_use,
            human_stereo_method=human_stereo_method,
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
        self.action_padding_ratio = action_padding_ratio
        self.repeat_frame_prob = repeat_frame_prob
        self.max_duration = max_duration
        self.num_demo_use = num_demo_use
        self.sampler = sampler
        self.temporally_independent_normalization = temporally_independent_normalization
        self.threadpool_limits_is_applied = False
        self.use_instruction = use_instruction
        self.text_feature_cache_dir = text_feature_cache_dir
        self.real_instruction_list = real_instruction_list
        self.device = None
        self.clip_model = None 
        self.tokenizer = None
        self.text_feature_bookmark = dict()
        self.text_feature_bookmark_max_size = 100000         # max bookmark buffer size: 100k
        self.human_stereo_method=human_stereo_method
        self.real_name_list = real_name_list
        self.real_sampler_idx = real_sampler_idx

    def get_dataset_name_list(self):
        return self.real_name_list
    
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = HRASequenceSamplerWrapper(
            alpha=self.alpha,
            shape_meta=self.shape_meta,
            replay_buffer_list=self.replay_buffer_list,
            human_replay_buffer_list=self.human_replay_buffer_list,
            real_sampler_idx=self.sampler.real_sampler_idx,
            real_instruction_list=self.real_instruction_list,
            rgb_keys=self.rgb_keys,
            lowdim_keys=self.sampler_lowdim_keys,
            key_horizon=self.key_horizon,
            key_latency_steps=self.key_latency_steps,
            key_down_sample_steps=self.key_down_sample_steps,
            robot_action_dim=self.robot_action_dim,
            gripper_action_dim=self.gripper_action_dim,
            episode_mask=self.val_mask,
            action_padding_ratio=self.action_padding_ratio,
            repeat_frame_prob=self.repeat_frame_prob,
            max_duration=self.max_duration,
            num_demo_use=self.num_demo_use,
            human_stereo_method=self.human_stereo_method,
        )
        val_set.val_mask = ~self.val_mask
        return val_set
    
    def set_device(self, device):
        self.device = device

    def get_text_feature(self, text):
        # we use cache and bookmark to speed up the text feature extraction

        text = text.replace('_', ' ').strip()
        if not text.endswith('.'):
            text = text + '.'

        if text in self.text_feature_bookmark:
            return self.text_feature_bookmark[text]
        
        
        if len(self.text_feature_bookmark) < self.text_feature_bookmark_max_size:
            text_filename = text.replace(" ", "_") 
            text_filename = hashlib.sha256(text_filename.encode('utf-8')).hexdigest()
            if len(self.text_feature_cache_dir) > 0:
                text_fp = os.path.join(self.text_feature_cache_dir, text_filename + '.npy')
                if os.path.exists(text_fp) and (not os.path.exists(text_fp + '.lock')):
                    text_feature = np.load(text_fp)
                    self.text_feature_bookmark[text] = text_feature
                    return text_feature
            if self.clip_model is None:
                import clip
                self.clip_model, _ = clip.load("ViT-B/16", device=self.device, jit=False)
                self.tokenizer = clip.tokenize
            text_tokens = self.tokenizer([text]).to(self.device)  
            text_features = self.clip_model.encode_text(text_tokens).detach().cpu().numpy()  # (1, 512)
            text_features = text_features[0]
            self.text_feature_bookmark[text] = text_feature

            if len(self.text_feature_cache_dir) > 0:
                with FileLock(text_fp + '.lock'):
                    if not os.path.exists(text_fp):
                        np.save(text_fp, text_features)
                        print(f'Saved text feature for "{text}" to {text_fp}')
                if os.path.exists(text_fp + '.lock'):
                    os.remove(text_fp + '.lock')

        else:
            # we do calculation if the bookmark is full.
            if self.clip_model is None:
                import clip
                self.clip_model, _ = clip.load("ViT-B/16", device=self.device, jit=False)
                self.tokenizer = clip.tokenize
            text_tokens = self.tokenizer([text]).to(self.device)  
            text_features = self.clip_model.encode_text(text_tokens).detach().cpu().numpy()  # (1, 512)
            text_features = text_features[0]
        return text_features
    

    def _get_normalizer(self, data_cache, use_ts_normalizer=False) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        if len(data_cache) == 0:
            return normalizer

        for key in data_cache.keys():
            data_cache[key] = np.concatenate(data_cache[key])
            # assert data_cache[key].shape[0] == len(self.sampler)
            assert len(data_cache[key].shape) == 3
            B, T, D = data_cache[key].shape
            if use_ts_normalizer:
                if key == 'action':
                    # use temporally dependent normalization
                    data_cache[key] = data_cache[key].reshape(B, T, D).transpose(0, 2, 1)  # B, D, T
                else:
                    # use temporally dependent normalization
                    data_cache[key] = data_cache[key].reshape(B, T * D)
            else:
                # use temporally independent normalization
                data_cache[key] = data_cache[key].reshape(B * T, D)

            # action
        action_normalizers = list()
        st_action_dim = 0 
        if use_ts_normalizer:
            B, D, T = data_cache['action'].shape
            data_cache['action'] = data_cache['action'].reshape(B, D * T)
            for idx, robot_dim in enumerate(self.robot_action_dim):
                assert robot_dim == 6
                action_normalizers.append(get_range_normalizer_from_stat(array_to_stats(data_cache['action'][..., st_action_dim * T: (st_action_dim + 3) * T])))  # pos
                action_normalizers.append(get_identity_normalizer_from_stat(array_to_stats(data_cache['action'][..., (st_action_dim + 3) * T: (st_action_dim + 9) * T]))) # rot, attention that it is 6d-rep from batch ! 
                st_action_dim = st_action_dim + 9             # watch out ! we use 9 dof action representation
                gripper_dim = self.gripper_action_dim[idx]
                action_normalizers.append(get_range_normalizer_from_stat(array_to_stats(data_cache['action'][..., st_action_dim * T: (st_action_dim + gripper_dim) * T])))  # gripper
                st_action_dim = st_action_dim + gripper_dim
        else:
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
            if key.endswith('pos') or 'pos_wrt' in key:
                stat = array_to_stats(data_cache[key])
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('pos_abs'):
                stat = array_to_stats(data_cache[key])
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('rot_axis_angle') or 'rot_axis_angle_wrt' in key:
                stat = array_to_stats(data_cache[key])
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('gripper_width'):
                stat = array_to_stats(data_cache[key])
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('gripper_pose'):
                stat = array_to_stats(data_cache[key])
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('gripper_force'):
                stat = array_to_stats(data_cache[key])
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('task_id'):
                stat = array_to_stats(data_cache[key])
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('instruction'):
                stat = array_to_stats(data_cache[key])
                this_normalizer = get_identity_normalizer_from_stat(stat)
            else:
                raise RuntimeError('unsupported')
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_identity_normalizer()
        
        return normalizer

    
    def get_normalizer(self, use_embodiment_normalizer=False, use_ts_normalizer=False, **kwargs) -> LinearNormalizer:

        # enumerate the dataset and save low_dim data
        self.sampler.ignore_rgb(True)
        batch_size = 32
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
            return self._get_normalizer(data_cache, use_ts_normalizer), self._get_normalizer(data_cache_human, use_ts_normalizer)
        else:
            return self._get_normalizer(data_cache, use_ts_normalizer), None

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
            if 'rgb' in key:
                if not key in data:
                    continue
                # move channel last to channel first
                # T,H,W,C
                # convert uint8 image to float32
                obs_dict[key] = np.moveaxis(data[key], -1, 1).astype(np.float32) / 255.
                # T,C,H,W
            elif 'pointcloud' in key:
                if not key in data:   # for ignore-key normalization stat setting
                    continue
                obs_dict[key] = data[key].astype(np.float32)
            else:
                raise RuntimeError(f'Unsupported key {key} in rgb_keys, please check the dataset shape meta.')
            del data[key]
        for key in self.sampler_lowdim_keys:
            obs_dict[key] = data[key].astype(np.float32)
            del data[key]

        del_keys = list()
        for key in obs_dict:
            if key.endswith('_demo_start_pose') or key.endswith('_demo_end_pose'):
                del_keys.append(key)
        for key in del_keys:
            del obs_dict[key]

        actions = list()
        for robot_id in range(self.num_robot):
            if f'robot{robot_id}_eef_pos' not in obs_dict:
                pass 
            else:
                # convert pose to mat
                pose_mat = pose_to_mat(np.concatenate([
                    obs_dict[f'robot{robot_id}_eef_pos'],
                    obs_dict[f'robot{robot_id}_eef_rot_axis_angle']
                ], axis=-1))
                # solve relative obs
                obs_pose_mat = convert_pose_mat_rep(
                    pose_mat, 
                    base_pose_mat=pose_mat[-1],
                    pose_rep=self.obs_pose_repr,
                    backward=False)
                # convert pose to pos + rot6d representation
                obs_pose = mat_to_pose10d(obs_pose_mat)
                # generate data
                obs_dict[f'robot{robot_id}_eef_pos'] = obs_pose[:,:3]
                obs_dict[f'robot{robot_id}_eef_rot_axis_angle'] = obs_pose[:,3:]

            # if f'robot{robot_id}_gripper_pose' not in obs_dict:
            #     pass 
            # else:
            #     if self.obs_pose_repr == 'relative':
            #         gripper_pose = obs_dict[f'robot{robot_id}_gripper_pose']
            #         obs_dict[f'robot{robot_id}_gripper_pose'] = gripper_pose - gripper_pose[-1]

            action_mat = pose_to_mat(data['action'][...,12 * robot_id: 12 * robot_id + 6])
            if f'robot{robot_id}_eef_pos' not in obs_dict:
                action_base_mat = action_mat[0]
            else:
                action_base_mat = pose_mat[-1]
            action_pose_mat = convert_pose_mat_rep(
                action_mat, 
                base_pose_mat=action_base_mat,
                pose_rep=self.action_pose_repr,
                backward=False)
        
            action_pose = mat_to_pose10d(action_pose_mat)
            actions.append(action_pose)

            gripper_action = data['action'][..., 12 * robot_id + 6: 12 * robot_id + 12]
            actions.append(gripper_action)

        if self.use_instruction:
            assert 'instruction' in data.keys(), 'use_instruction is True, but instruction is not in data.'
            obs_dict['instruction'] = self.get_text_feature(data['instruction'])
        
        data['action'] = np.concatenate(actions, axis=-1)   
        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(data['action'].astype(np.float32)),
            'alpha': torch.tensor(data['alpha'], dtype=torch.float32),
            'embodiment': torch.tensor(data['embodiment'], dtype=torch.float32),
        }
        return torch_data
