if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import pickle
import tqdm
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.dataset.base_dataset import BaseImageDataset, BaseDataset
from common.checkpoint_util import TopKCheckpointManager
from common.json_logger import JsonLogger
from common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.common.lr_decay import param_groups_lrd
from accelerate import Accelerator
from diffusion_policy.policy.embodiment_adversarial_network import EmbodimentAdversarialNetwork

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionUnetImageWorkspaceCheck(BaseWorkspace):
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.shape_meta = cfg.shape_meta

        # configure model
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        # self.optimizer = hydra.utils.instantiate(
        #     cfg.optimizer, params=self.model.parameters())
        if cfg.training.layer_decay < 1.0:
            assert not cfg.policy.obs_encoder.use_lora
            assert not cfg.policy.obs_encoder.share_rgb_model
            obs_encorder_param_groups = param_groups_lrd(self.model.obs_encoder,
                                                         shape_meta=cfg.shape_meta,
                                                         weight_decay=cfg.optimizer.encoder_weight_decay,
                                                         no_weight_decay_list=self.model.obs_encoder.no_weight_decay(),
                                                         layer_decay=cfg.training.layer_decay)
            count = 0
            for group in obs_encorder_param_groups:
                count += len(group['params'])
            if cfg.policy.obs_encoder.feature_aggregation == 'map':
                obs_encorder_param_groups.extend([{'params': self.model.obs_encoder.attn_pool.parameters()}])
                for _ in self.model.obs_encoder.attn_pool.parameters():
                    count += 1
            print(f'obs_encorder params: {count}')
            param_groups = [{'params': self.model.model.parameters()}]
            param_groups.extend(obs_encorder_param_groups)
        else:
            obs_encorder_lr = cfg.optimizer.lr
            if cfg.policy.obs_encoder.pretrained and not cfg.policy.obs_encoder.use_lora:
                obs_encorder_lr *= cfg.training.encoder_lr_coefficient
                print('==> reduce pretrained obs_encorder\'s lr')
            obs_encorder_params = list()
            for param in self.model.obs_encoder.parameters():
                if param.requires_grad:
                    obs_encorder_params.append(param)
            print(f'obs_encorder params: {len(obs_encorder_params)}')
            param_groups = [
                {'params': self.model.model.parameters()},
                {'params': obs_encorder_params, 'lr': obs_encorder_lr}
            ]
        optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
        optimizer_cfg.pop('_target_')
        if 'encoder_weight_decay' in optimizer_cfg.keys():
            optimizer_cfg.pop('encoder_weight_decay')
        self.optimizer = torch.optim.AdamW(
            params=param_groups,
            **optimizer_cfg
        )

        # configure training state
        self.global_step = 0
        self.epoch = 0

        # do not save optimizer if resume=False
        if not cfg.training.resume:
            self.exclude_keys = ['optimizer']

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        cfg.task.dataset.pose_repr = cfg.task.pose_repr
        if hasattr(cfg.task, "dataset_hra3"):
            cfg.task.dataset_hra3.pose_repr = cfg.task.pose_repr

        accelerator = Accelerator(log_with='wandb')
        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        wandb_cfg.pop('project')
        accelerator.init_trackers(
            project_name=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": wandb_cfg}
        )

        # resume training
        if True:
            lastest_ckpt_path = self.get_checkpoint_path()
            accelerator.print(f"Resuming from checkpoint {lastest_ckpt_path}")
            if lastest_ckpt_path.is_file():
                self.load_checkpoint(path=lastest_ckpt_path)
                # try:
                #     self.load_checkpoint(path=lastest_ckpt_path)
                # except:
                #     print("Checking HRA3 Algorithm")
                #     self.obs_feature_dim = self.model.obs_adv_feature_dim
                #     self.embodiment_adversarial_network = EmbodimentAdversarialNetwork(self.obs_feature_dim, 
                #                                                            down_dims=cfg.embodiment_adversarial.nn_down_dims,
                #                                                            kernel_size=cfg.embodiment_adversarial.nn_kernel_size,
                #                                                            n_groups=cfg.embodiment_adversarial.nn_n_groups
                #                                                            )              
                #     self.optimizer_adversarial = torch.optim.AdamW(params=self.embodiment_adversarial_network.parameters(),lr=1e-4)
                #     self.load_checkpoint(path=lastest_ckpt_path)
            else:
                raise ValueError(f"{lastest_ckpt_path} not found.")

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        # data = dataset.__getitem__(100)
        # import pdb; pdb.set_trace()
        assert isinstance(dataset, BaseImageDataset) or isinstance(dataset, BaseDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)

        # load normalizer on all processes
        accelerator.wait_for_everyone()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        
        print('train dataset:', len(dataset), 'train dataloader:', len(train_dataloader))
        print('val dataset:', len(val_dataset), 'val dataloader:', len(val_dataloader))

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1 if not cfg.training.resume else -1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # # configure logging
        # wandb_run = wandb.init(
        #     dir=str(self.output_dir),
        #     config=OmegaConf.to_container(cfg, resolve=True),
        #     **cfg.logging
        # )
        # wandb.config.update(
        #     {
        #         "output_dir": self.output_dir,
        #     }
        # )

        # configure checkpoint
        if cfg.checkpoint.only_save_recent:
            cfg.checkpoint.topk.monitor_key = 'epoch'
            cfg.checkpoint.topk.mode = 'max'
        else:
            cfg.checkpoint.topk.monitor_key = 'val_action_mse_error'
            assert cfg.training.checkpoint_every >= cfg.training.sample_every and cfg.training.checkpoint_every % cfg.training.sample_every == 0
        cfg.checkpoint.topk.format_str = 'epoch-{epoch:04d}-' + cfg.checkpoint.topk.monitor_key + '-{' + cfg.checkpoint.topk.monitor_key + ':.5f}.ckpt'
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        # device = torch.device(cfg.training.device)
        # self.model.to(device)
        # if self.ema_model is not None:
        #     self.ema_model.to(device)
        # optimizer_to(self.optimizer, device)

        # accelerator
        train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler = accelerator.prepare(
            train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler
        )

        device = self.model.device
        if self.ema_model is not None:
            self.ema_model.to(device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 10
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                self.model.train()

                step_log = dict()
                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)


                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer

                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        
                        # always use the latest batch
                        train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.model(batch)
                        if type(raw_loss) is tuple:
                            raw_loss, raw_feat = raw_loss
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        accelerator.backward(loss)
                        
                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train/loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'train/epoch': self.epoch,
                            'train/lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            accelerator.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break
                        break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train/loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = accelerator.unwrap_model(self.model)
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run validation
                if (self.epoch % cfg.training.val_every) == 0 and len(val_dataloader) > 0:
                    # with torch.no_grad():  # Not used due to PointNeXt Conflict
                    val_losses = list()
                    with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                            leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            loss = self.model(batch)
                            val_losses.append(loss.item())
                            if (cfg.training.max_val_steps is not None) \
                                and batch_idx >= (cfg.training.max_val_steps-1):
                                break
                    if len(val_losses) > 0  and accelerator.is_main_process:
                        val_loss = np.mean(val_losses)
                        # log epoch average validation loss
                        step_log['val_loss'] = val_loss

                def log_action_mse(step_log, category, pred_action, gt_action):
                    B, T, _ = pred_action.shape
                    pred_action = pred_action.view(B, T, -1, self.shape_meta['action']['shape'][0])
                    gt_action = gt_action.view(B, T, -1, self.shape_meta['action']['shape'][0])
                    step_log[f'{category}/action_mse_error'] = torch.nn.functional.mse_loss(pred_action, gt_action)
                    step_log[f'{category}_action_mse_error_pos'] = torch.nn.functional.mse_loss(pred_action[..., :3], gt_action[..., :3])
                    step_log[f'{category}_action_mse_error_rot'] = torch.nn.functional.mse_loss(pred_action[..., 3:9], gt_action[..., 3:9])
                    if gt_action.shape[-1] > 9:
                        step_log[f'{category}_action_mse_error_hand'] = torch.nn.functional.mse_loss(pred_action[..., 9:], gt_action[..., 9:])
                    else:
                        step_log[f'{category}_action_mse_error_hand'] = torch.Tensor([0.0]).to(device)
                    return step_log

                def cal_action_mse(pred_action, gt_action):
                    B, T, _ = pred_action.shape
                    pred_action = pred_action.view(B, T, -1, self.shape_meta['action']['shape'][0])
                    gt_action = gt_action.view(B, T, -1, self.shape_meta['action']['shape'][0])
                    action_mse_error = torch.nn.functional.mse_loss(pred_action, gt_action)
                    action_mse_error_pos = torch.nn.functional.mse_loss(pred_action[..., :3], gt_action[..., :3])
                    action_mse_error_rot = torch.nn.functional.mse_loss(pred_action[..., 3:9], gt_action[..., 3:9])
                    if gt_action.shape[-1] > 9:                   
                        action_mse_error_width = torch.nn.functional.mse_loss(pred_action[..., 9:], gt_action[..., 9:])
                    else:
                        action_mse_error_width = torch.Tensor([0.0]).to(device)
                    return action_mse_error, action_mse_error_pos, action_mse_error_rot, action_mse_error_width

                # run diffusion sampling on a training batch
                if True:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        import pdb; pdb.set_trace()
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        gt_action = batch['action']
                        pred_action = policy.predict_action(batch['obs'], None, embodiment=batch['embodiment'])['action_pred']
                        pred_action_model = self.model.predict_action(batch['obs'], None, embodiment=batch['embodiment'])['action_pred']
                        all_preds, all_gt = accelerator.gather_for_metrics((pred_action, gt_action))
                        step_log = log_action_mse(step_log, 'train', all_preds, all_gt)

                        if len(val_dataloader) > 0:
                            action_mse_error_list, action_mse_error_pos_list, action_mse_error_rot_list, action_mse_error_width_list = [], [], [], []
                            for idx, val_sampling_batch in tqdm.tqdm(enumerate(val_dataloader)):
                                batch = dict_apply(val_sampling_batch, lambda x: x.to(device, non_blocking=True))
                                # Image.fromarray((val_sampling_batch['obs']['camera0_rgb'][0,0]*255).detach().cpu().numpy().astype(np.uint8).transpose(1,2,0)).save('test.png')
                                gt_action = batch['action']
                                pred_action = policy.predict_action(batch['obs'], None, embodiment=batch['embodiment'])['action_pred']
                                all_preds, all_gt = accelerator.gather_for_metrics((pred_action, gt_action))
                                action_mse_error, action_mse_error_pos, action_mse_error_rot, action_mse_error_width = \
                                    cal_action_mse(all_preds, all_gt)
                                # import pdb; pdb.set_trace()
                                # export batch 
                                batch['action'] = all_preds
                                with open(os.path.join(self.output_dir, f'val_batch_{idx}.pkl'), 'wb') as f:
                                    pickle.dump(batch, f)
                                pickle.load(open(os.path.join(self.output_dir, f'val_batch_{idx}.pkl'), 'rb'))
                                
                                # import pdb; pdb.set_trace()

                                action_mse_error_list.append(action_mse_error)
                                action_mse_error_pos_list.append(action_mse_error_pos)
                                action_mse_error_rot_list.append(action_mse_error_rot)
                                action_mse_error_width_list.append(action_mse_error_width)
                            step_log['val/action_mse_error'] = torch.mean(torch.stack(action_mse_error_list)).item()
                            step_log['val_action_mse_error_pos'] = torch.mean(torch.stack(action_mse_error_pos_list)).item()
                            step_log['val_action_mse_error_rot'] = torch.mean(torch.stack(action_mse_error_rot_list)).item()
                            step_log['val_action_mse_error_hand'] = torch.mean(torch.stack(action_mse_error_width_list)).item()
                            import pdb; pdb.set_trace()
                            
                        del batch
                        del gt_action
                        del pred_action
                
                self.optimizer.zero_grad()
                accelerator.wait_for_everyone()
                    
                # checkpoint
                if cfg.checkpoint.only_save_recent:
                    if ((self.epoch % cfg.training.checkpoint_every) == 0 or \
                         self.epoch == cfg.training.num_epochs - 1) and accelerator.is_main_process:
                        if_save_ckpt = True
                    else:
                        if_save_ckpt = False
                elif ((self.epoch % cfg.training.checkpoint_every) == 0 or self.epoch == cfg.training.num_epochs - 1) and accelerator.is_main_process:
                    if_save_ckpt = True
                else:
                    if_save_ckpt = False
                if if_save_ckpt:

                    # unwrap the model to save ckpt
                    model_ddp = self.model
                    self.model = accelerator.unwrap_model(self.model)

                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        if new_key == 'train_epoch':
                            new_key = 'epoch'
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    if cfg.checkpoint.only_save_recent or (cfg.training.num_epochs - self.epoch) in [5]:
                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                        if topk_ckpt_path is not None:
                            self.save_checkpoint(path=topk_ckpt_path)
                    else:
                        raise NotImplementedError("Only 'only_save_recent' checkpoint is supported")

                    # recover the DDP model
                    self.model = model_ddp
                # ========= eval end for this epoch ==========
                # end of epoch
                # log of last step is combined with validation and rollout
                accelerator.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

        accelerator.end_training()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
