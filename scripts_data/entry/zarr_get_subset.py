
import os
import click
import json
import shutil
import numpy as np
from tqdm import tqdm
from common.replay_buffer import ReplayBuffer


def get_replay_buffer_list(dataset_path):
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
    return dataset_path_list


def get_instruction_from_filename_list(filename):
    if '+' in filename and filename.find('+') > 0:
        instruction = filename[filename.find('+') + 1:filename.rfind('+')]
        instruction = instruction.strip()
        instruction = instruction.replace('_', ' ')
        if not instruction.endswith('.'):
            instruction += '.'
        if '+' in instruction:
            raise ValueError(f'Filename {filename} contains multiple instructions between + signs.')
        if len(instruction) > 0:
            pass
        else:
            raise ValueError(f'Filename {filename} contains empty instruction between + signs.')
    else:
        raise ValueError(f'Filename {filename} does not contain instruction in +task_name+ format.')
    return instruction


@click.command()
@click.option('--dataset_path', '-i', required=True)
@click.option('--ratio', '-r', type=float, required=True)
def main(dataset_path, ratio):

    if '.' in dataset_path:
        raise ValueError(f'Please do not use file path, only use folder path, got {dataset_path}')
    zarr_list = os.listdir(dataset_path)
    dataset_path_new = dataset_path + f"_subset{ratio}"
    os.makedirs(dataset_path_new, exist_ok=True)

    zarr_list = [zarr_fp for zarr_fp in zarr_list if zarr_fp.endswith('.zarr')]
    n_task = len(zarr_list)
    n_episodes_list = []
    dataset_meta = dict()
    for i in range(n_task):
        zarr_name = zarr_list[i]
        task_name = get_instruction_from_filename_list(zarr_name)
        replay_buffer = ReplayBuffer.create_from_path(os.path.join(dataset_path, zarr_name), mode='r')
        n_episodes_list.append(replay_buffer.n_episodes)
    
    n_episodes = np.array(n_episodes_list)
    n_episodes_remove = np.zeros_like(n_episodes, dtype=np.int32)
    n_episodes_total = n_episodes.sum()
    n_episodes_target = int(ratio * n_episodes_total)
    while n_episodes_total > n_episodes_target:
        n_episodes_max = n_episodes.max()
        idx_max = np.argmax(n_episodes)
        n_episodes_remove[idx_max] += 1
        n_episodes[idx_max] -= 1
        n_episodes_total = n_episodes.sum()

    for i in tqdm(range(n_task)):
        zarr_name = zarr_list[i]
        replay_buffer = ReplayBuffer.create_from_path(os.path.join(dataset_path, zarr_name), mode='r')
        replay_buffer_new = ReplayBuffer.create_from_path(os.path.join(dataset_path_new, zarr_name), mode='w')
        n_episodes = replay_buffer.n_episodes
        episode_choose_idx = np.random.choice(n_episodes, n_episodes - n_episodes_remove[i], replace=False)
        for episode_idx in episode_choose_idx:
            episode_data = replay_buffer.get_episode(episode_idx)
            replay_buffer_new.add_episode(episode_data, compressors='disk')



if __name__ == '__main__':
    main() 