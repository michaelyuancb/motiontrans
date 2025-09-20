
import os
import click
import json
import shutil
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
@click.option('--input_dir', '-i', required=True)
@click.option('--output_dir', '-o', required=True)
def main(input_dir, output_dir):
    dataset_path_list = input_dir.split("|")
    dataset_path_list = [dataset for dataset in dataset_path_list if dataset is not None and len(dataset) > 0]
    metadata_dataset = dict()
    for dataset_path in tqdm(dataset_path_list, "Datasets"):
        if '.zarr' in dataset_path:
            raise ValueError(f'Please do not use file path, only use folder path, got {dataset_path}')
        zarr_list = os.listdir(dataset_path)
        zarr_list = [zarr_fp for zarr_fp in zarr_list if zarr_fp.endswith('.zarr')]
        n_task = len(zarr_list)
        n_episodes_total = 0
        n_frames_total = 0
        task_list = []
        dataset_meta = dict()
        for i in range(n_task):
            zarr_name = zarr_list[i]
            task_name = get_instruction_from_filename_list(zarr_name)
            replay_buffer = ReplayBuffer.create_from_path(os.path.join(dataset_path, zarr_name), mode='r')
            n_episodes = replay_buffer.n_episodes
            if n_episodes == 0:
                shutil.rmtree(os.path.join(dataset_path, zarr_name))
                continue
            n_frames = len(replay_buffer.data['robot0_eef_pos'])
            dataset_meta[task_name] = {"n_episodes": n_episodes, "n_frames": n_frames, "instruction": task_name}
            task_list.append(task_name)
            n_episodes_total += n_episodes
            n_frames_total += n_frames
        metadata_dataset[dataset_path] = {
            "n_tasks": n_task,
            "n_episodes": n_episodes_total,
            "n_frames": n_frames_total,
            "task_list": task_list,
            "dataset_meta": dataset_meta
        }
    n_tasks_total = 0
    n_episodes_total = 0
    n_frames_total = 0
    task_list_total = []
    conclude_meta = dict()
    for dataset_path, meta in metadata_dataset.items():
        n_tasks_total += meta['n_tasks']
        n_episodes_total += meta['n_episodes']
        n_frames_total += meta['n_frames']
        conclude_meta[dataset_path] = {"dataset": dataset_path.split('/')[-1], "n_tasks": meta['n_tasks'], "n_episodes": meta['n_episodes'], "n_frames": meta['n_frames']}
        task_list_total = task_list_total + meta['task_list']

    metadata = dict()
    metadata['n_tasks'] = n_tasks_total
    metadata['n_episodes'] = n_episodes_total
    metadata['n_frames'] = n_frames_total
    metadata['datasets_info'] = conclude_meta
    metadata['input_str'] = input_dir
    metadata['task_list'] = task_list_total
    metadata['datasets_details'] = metadata_dataset

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)



if __name__ == '__main__':
    main() 