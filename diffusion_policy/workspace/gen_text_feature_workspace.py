if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
from omegaconf import OmegaConf
import pathlib
import copy
from tqdm import tqdm
import numpy as np
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.dataset.base_dataset import BaseImageDataset
import hashlib
from threadpoolctl import threadpool_limits
from tqdm import trange, tqdm
from filelock import FileLock

OmegaConf.register_new_resolver("eval", eval, replace=True)
import clip
clip_model, _ = clip.load("ViT-B/16", device='cuda', jit=False)
tokenizer = clip.tokenize


def get_text_feature(text, text_feature_cache_dir, verbose=False):
    # we use cache and bookmark to speed up the text feature extraction

    text = text.replace('_', ' ').strip()
    if not text.endswith('.'):
        text = text + '.'
    
    text_filename = text.replace(" ", "_") 
    text_filename = hashlib.sha256(text_filename.encode('utf-8')).hexdigest()
    text_fp = os.path.join(text_feature_cache_dir, text_filename + '.npy')
    text_tokens = tokenizer([text]).cuda()
    text_features = clip_model.encode_text(text_tokens).detach().cpu().numpy()  # (1, 512)
    text_features = text_features[0]
    np.save(text_fp, text_features)
    if verbose:
        print(f'Saved text feature for "{text}" to {text_fp}')


class GenerateTextFeatureWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)


    def run(self):
        cfg = copy.deepcopy(self.cfg)
        cfg.task.dataset.pose_repr = cfg.task.pose_repr

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        text_feature_cache_dir = dataset.text_feature_cache_dir
        sampler = dataset.sampler
        instruction_list = []
        n_data = len(sampler)
        for i in tqdm(range(n_data), desc="sample_instruction"):
            instruction_list.append(sampler.sample_instruction(i))
        instruction_list = list(set(instruction_list))
        for instruction in tqdm(instruction_list, desc="generate_feature"):
            get_text_feature(instruction, text_feature_cache_dir, verbose=False)

        print("Fininsh.")