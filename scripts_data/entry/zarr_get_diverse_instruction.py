import zarr
import numpy as np
import requests
from PIL import Image
import io
import click
import argparse
import os
import json
import shutil
import base64
from tqdm import tqdm
from common.replay_buffer import ReplayBuffer
from scripts_data.entry.zarr_get_metadata import get_instruction_from_filename_list
from openai import AzureOpenAI

PROMPT_HEAD = """
You are a human manipulation task instruction re-writer. Given \\
1. an original task instruction \\
2. three image from human execution video of this task \\
your goal is to re-write 20 new instructions which: \\
1. the task should be exactly the same with the original instruction. \\
2. you may make more details for some instructions with images hint, e.g. which color/material/type of objects, under which backgrounds, detailed actions, use which hand, etc. \\
3. keep the diversity of your re-written instructions, e.g. some is short and concise (even only serveral words), some is long and detailed, some is with more details on objects, some is with more details on actions, etc. \\
your output format should be like this: \\
[1] new_instruction_1. \\
[2] new_instruction_2. \\
[3] new_instruction_3. \\
... \\
[20] new_instruction_20. \\
Please follow the format strictly, do not add any other text and content. Make sure the task is the same for new and original instruction, and follows the content of images strictly. \\
Input images: 
"""

def get_response(client, prompt, model='gpt-4o-mini'):
    n_trial = 3
    response = None
    for _ in range(n_trial):
        try:
            response = client.chat.completions.create(
                messages=prompt,
                max_tokens=4096,
                temperature=0.7,
                top_p=1.0,
                model=model
            )
            break
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            continue
    if response is None:
        return None
    return response.choices[0].message.content


def img_to_base64(img_array):
    """Convert numpy image array to base64 string"""
    img = Image.fromarray(img_array)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def get_new_episode_list(zarr_name, episode_idx, client, episode, instruction_old, embodiment):

    images = episode['camera0_rgb']
    n_frames = len(images)
    image_idx_list = np.random.choice(n_frames, 1, replace=False)
    images = [images[i] for i in image_idx_list]

    # for i, image in enumerate(images):
    #     Image.fromarray(image).save(f'image_{i}.png')

    # Prepare the prompt for the AI model
    chat_prompt = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": PROMPT_HEAD.replace("human", embodiment)
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_to_base64(images[0])}",
                    }
                },
                {
                    "type": "text",
                    "text": "Original instruction: " + instruction_old + ". Output:"
                }
            ]
        }
    ]

    # Get the response from the AI model
    response = get_response(client, chat_prompt, model='gpt-4o')
    if response is None:
        print(f"Failed to get response for episode {episode_idx} in {zarr_name}.")
        instructions = [instruction_old.ljust(200)]
    else:
        # get the instructions list from the response
        instructions = response.split('\n')
        instructions = [inst.strip() for inst in instructions if inst.strip()]
        instructions = [inst.split('] ')[-1].strip() for inst in instructions]
        # fill each instruction to length of 200
        instructions = [inst.ljust(200) for inst in instructions]
    instruction_old = instruction_old.ljust(200)

    instruction_list = []
    for i in range(n_frames):
        random_idx = np.random.randint(4)
        if random_idx == 0:
            instruction_list.append(instruction_old)
        else:
            instruction_new = instructions[np.random.randint(len(instructions))]
            instruction_list.append(instruction_new)
        
    episode['instruction'] = np.array(instruction_list)
    return episode, instructions



@click.command()
@click.option('--input_dir', '-i', required=True)
@click.option('--output_dir', '-o', required=True)
@click.option('--embodiment', '-e', default='human', help='Embodiment type, e.g., human, robot')
def main(input_dir, output_dir, embodiment):
    # Configure OpenRouter API
    client = AzureOpenAI(
        azure_endpoint=os.environ.get('AZURE_ENDPOINT'),
        api_key=os.environ.get('AZURE_API_KEY'),
        api_version="2024-12-01-preview",
    )

    dataset_path_list = input_dir.split("|")
    dataset_path_list = [dataset for dataset in dataset_path_list if dataset is not None and len(dataset) > 0]
    for dataset_path in tqdm(dataset_path_list, desc="Processing Zarr files"):
        if not os.path.exists(dataset_path):
            print(f"Zarr file {dataset_path} does not exist, skipping.")
            continue
        if '.' in dataset_path:
            raise ValueError(f'Please do not use file path, only use folder path, got {dataset_path}')
        dataset_name = dataset_path.split('/')[-1]
        zarr_list = os.listdir(dataset_path)
        zarr_list = [zarr_fp for zarr_fp in zarr_list if zarr_fp.endswith('.zarr')]
        n_task = len(zarr_list)
        for i in range(n_task):
            zarr_name = zarr_list[i]
            task_name = get_instruction_from_filename_list(zarr_name)
            zarr_name_new = zarr_name.replace(".zarr", "gpt.zarr")
            meta_path = os.path.join(output_dir, dataset_name, f'{zarr_name.replace(".zarr","")}_gpt.json')
            if os.path.exists(meta_path):
                continue
            replay_buffer_old = ReplayBuffer.create_from_path(os.path.join(dataset_path, zarr_name), mode='r')
            new_buffer_path = os.path.join(output_dir, dataset_name, zarr_name_new)
            if os.path.exists(new_buffer_path):
                shutil.rmtree(new_buffer_path)
            replay_buffer_new = ReplayBuffer.create_from_path(new_buffer_path, mode='w')
            n_episodes = replay_buffer_old.n_episodes
            if n_episodes == 0:
                shutil.rmtree(os.path.join(dataset_path, zarr_name))
                continue
            dataset_meta= dict()
            dataset_meta['zarr_name'] = zarr_name
            dataset_meta['task_name'] = task_name
            dataset_meta['n_episodes'] = n_episodes
            for j in tqdm(range(n_episodes), desc=f"Processing episodes in {zarr_name}"):
                episode = replay_buffer_old.get_episode(j)
                if 'egodex' not in dataset_name:
                    instruction_old = task_name
                    if not instruction_old.endswith('.'):
                        instruction_old += '.'
                    episode_new, instruction_new = get_new_episode_list(zarr_name, j, client, episode, instruction_old, embodiment)
                    dataset_meta['episode_' + str(j)] = {
                        "instruction_old": instruction_old,
                        "instruction_new": instruction_new
                    }
                    replay_buffer_new.add_episode(episode_new, compressors='disk')

            with open(meta_path, 'w') as f:
                json.dump(dataset_meta, f, indent=4)
            


if __name__ == '__main__':
    main() 
    