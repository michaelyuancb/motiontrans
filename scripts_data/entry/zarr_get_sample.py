import os
import time
import torch
import cv2
import imageio
import torch
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
import click
import pathlib
from common.replay_buffer import ReplayBuffer
from tqdm import tqdm

# Import roboengine modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

@click.command()
@click.option('--replay_buffer_fp', '-i',  required=True)
@click.option('--output_dir', '-o', required=True)
@click.option('--num_use_demo', '-n', default=1)
def main(replay_buffer_fp, output_dir, num_use_demo):


    output_fp = os.path.join(output_dir, "demo.zarr")

    output_fp = Path(output_fp)
    if output_fp.exists():
        shutil.rmtree(output_fp)

    replay_buffer_new = ReplayBuffer.create_from_path(output_fp, mode='w')
    
    replay_buffer_fp = Path(replay_buffer_fp)
    replay_buffer_org = ReplayBuffer.create_from_path(replay_buffer_fp, mode='r')

    n_episodes = replay_buffer_org.n_episodes
    if num_use_demo > 0 and num_use_demo < n_episodes:
        n_episodes = num_use_demo

    for episode_idx in tqdm(range(n_episodes), desc=f'Processing {replay_buffer_fp.name}'):
        episode_data = replay_buffer_org.get_episode(episode_idx)
        replay_buffer_new.add_episode(episode_data, compressors='disk')

    print("Saving to disk")


if __name__ == '__main__':
    main()