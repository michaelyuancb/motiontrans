"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
import click
import dill
import time
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

import os

@click.command()
@click.option('--ckpt_path', '-c', required=True, help='Path to checkpoint')
@click.option('--dataset_path', '-d', required=True, help='Path to dataset')
@click.option('--run_dir', required=True, type=str, help='Result directory')
@click.option('--num_demo_use', type=int, default=-1, help='Number of demonstrations to use for finetuning')
@click.option('--num_finetune_epochs', type=int, default=50, help='Result directory')
@click.option('--lr_warmup_steps', type=int, default=0, help='Result directory')
@click.option('--checkpoint_every', type=int, default=10, help='Result directory')
@click.option('--batch_size', type=int, default=32, help='Result directory')
@click.option('--lr', type=float, default=1e-4, help='Result directory')
@click.option('--freeze_encoder', is_flag=True, default=False, help='Freeze the encoder during finetuning')
@click.option('--from_scratch', is_flag=True, default=False, help='Finetune training from scratch')
def main(ckpt_path,
         dataset_path,
         run_dir,
         num_demo_use,
         num_finetune_epochs,
         lr_warmup_steps,
         checkpoint_every,
         batch_size,
         lr,
         freeze_encoder,
         from_scratch):
    
    # current hydra folder
    print(f"Current working directory: {os.getcwd()}")

    try:
        local_rank = os.environ["LOCAL_RANK"]
        print(f"LOCAL_RANK: {local_rank}")
    except:
        local_rank = "0" 

    # load checkpoint
    if not ckpt_path.endswith('.ckpt'):
        ckpt_path = os.path.join(ckpt_path, 'checkpoints', 'latest.ckpt')
    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']

    if not hasattr(cfg.training, 'resume'):
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        cfg_dict["training"]["resume"] = True
        cfg = OmegaConf.create(cfg_dict)
    cfg.training.resume = True 
    if not hasattr(cfg.training, 'ckpt_path'):
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        cfg_dict["training"]["ckpt_path"] = ""
        cfg = OmegaConf.create(cfg_dict)
    cfg.training.ckpt_path = ckpt_path

    cfg.task.dataset_path = dataset_path
    cfg.task.dataset.dataset_path = dataset_path
    if hasattr(cfg.task, 'dataset_hra3'):
        cfg.task.dataset_hra3.dataset_path = dataset_path

    cfg.task.human_dataset_path = None
    cfg.task.dataset.human_dataset_path = None
    if hasattr(cfg.task, 'dataset_hra3'):
        cfg.task.dataset_hra3.human_dataset_path = None

    cfg.task.alpha = 1.0
    cfg.task.dataset.alpha = 1.0
    if hasattr(cfg.task, 'dataset_hra3'):
        cfg.task.dataset_hra3.alpha = 1.0

    if not hasattr(cfg.task.dataset, 'num_demo_use'):
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        cfg_dict["task"]["dataset"]["num_demo_use"] = 0
        if hasattr(cfg.task, 'dataset_hra3'):
            cfg_dict["task"]["dataset_hra3"]["num_demo_use"] = 0
        cfg = OmegaConf.create(cfg_dict)
    cfg.task.dataset.num_demo_use = num_demo_use
    if hasattr(cfg.task, 'dataset_hra3'):
        cfg.task.dataset_hra3.num_demo_use = num_demo_use

    os.makedirs(run_dir, exist_ok=True)
    cfg.multi_run.run_dir = run_dir
    cfg.multi_run.wandb_name_base = run_dir
    cfg.logging.name = run_dir.split('/')[-1]

    cfg.task.dataset.val_ratio = 0
    if hasattr(cfg.task, 'dataset_hra3'):
        cfg.task.dataset_hra3.val_ratio = 0

    cfg.training.freeze_encoder = freeze_encoder
    cfg.training.num_epochs = num_finetune_epochs
    cfg.training.checkpoint_every = checkpoint_every
    cfg.training.val_every = 1
    cfg.training.sample_every = 1
    cfg.training.gradient_accumulate_every = 1
    cfg.training.lr_warmup_steps = lr_warmup_steps

    cfg.dataloader.batch_size = batch_size
    cfg.val_dataloader.batch_size = batch_size

    cfg.optimizer.lr = lr

    if hasattr(cfg, 'embodiment_adversarial'):
        cfg.embodiment_adversarial.judger_warmup_epochs = 0
        cfg.embodiment_adversarial.adv_update_num_per_step = -1

    # we save all checkpoints for finetuning
    cfg.checkpoint.topk.k = 1000000000000

    cls = hydra.utils.get_class(cfg._target_)

    workspace: BaseWorkspace = cls(cfg, output_dir=run_dir)

    workspace.set_finetune_mode(from_scratch)

    start_time = time.time()

    workspace.run()

    if local_rank == "0":
        # record finetune time
        use_time = time.time() - start_time
        hour_minute_sec_str = time.strftime("%H hours %M minutes %S seconds", time.gmtime(use_time))
        with open(os.path.join(run_dir, 'use_time.txt'), 'w') as f:
            f.write(hour_minute_sec_str)


if __name__ == "__main__":
    main()