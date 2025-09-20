"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

import os
# os.environ["WANDB_DISABLED"]="true"

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    # current hydra folder
    print(f"Current working directory: {os.getcwd()}")
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    try:
        local_rank = os.environ["LOCAL_RANK"]
        print(f"LOCAL_RANK: {local_rank}")
    except:
        local_rank = "0" 
    if local_rank == "0":
        cfg_save_fp = os.path.join(cfg.multi_run['run_dir'], "config.yaml")
        print(f"Saving config to {cfg_save_fp}")
        with open(cfg_save_fp, 'w') as f:
            OmegaConf.save(cfg, f)

    cls = hydra.utils.get_class(cfg._target_)

    if cfg.task.use_instruction:
        import clip 
        clip_model, clip_preprocess = clip.load('ViT-B/16', device="cpu", jit=False)

    workspace: BaseWorkspace = cls(cfg)

    workspace.run()

if __name__ == "__main__":
    main()