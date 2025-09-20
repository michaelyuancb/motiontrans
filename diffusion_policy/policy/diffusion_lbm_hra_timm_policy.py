from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.transformer_for_action_diffusion import TransformerForActionDiffusion
from diffusion_policy.model.vision.transformer_obs_encoder import TransformerObsEncoder


class DiffusionLargeBehaviourModelHRATimmPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: TransformerObsEncoder,
            num_inference_steps=None,
            input_pertub=0.1,
            diffusion_step_embed_dim=256,
            # arch
            n_layer=8,
            n_head=12,
            n_emb=768,
            use_adaptive_layernorm=True,
            use_adaptive_layerscale=True,
            use_embodiment_normalizer=False,
            use_ts_normalizer=False,
            # parameters passed to step
            **kwargs):
        super().__init__()

        self.use_embodiment_normalizer = use_embodiment_normalizer
        self.use_ts_normalizer = use_ts_normalizer

        self.ts_normalizer_obs_key = []
        self.ts_normalizer_shape_cache = dict()
        for key in shape_meta['obs'].keys():
            if shape_meta['obs'][key]['type'] == 'low_dim':
                self.ts_normalizer_obs_key.append(key)

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        action_horizon = shape_meta['action']['horizon']
        
        obs_shape = obs_encoder.output_shape()
        cond_dim = obs_shape[-1]
        print(f"LBM Condition Dim: ", cond_dim)
        
        model = TransformerForActionDiffusion(
            input_dim=action_dim,
            output_dim=action_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            cond_dim=cond_dim,
            action_horizon=action_horizon,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            use_adaptive_layernorm=use_adaptive_layernorm,
            use_adaptive_layerscale=use_adaptive_layerscale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.input_pertub = input_pertub
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            cond=None, generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor], embodiment: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input

        if self.use_ts_normalizer: 
            for key in self.ts_normalizer_obs_key:
                T, D = obs_dict[key].shape[-2:]
                if key not in self.ts_normalizer_shape_cache:
                    self.ts_normalizer_shape_cache[key] = (T, D)
                obs_dict[key] = obs_dict[key].reshape(obs_dict[key].shape[:-2] + (1, T * D,))
        nobs = self.normalizer.normalize(obs_dict)
        if self.use_ts_normalizer: 
            for key in self.ts_normalizer_obs_key:
                nobs[key] = nobs[key].reshape(
                    nobs[key].shape[:-2] + self.ts_normalizer_shape_cache[key]
                )
        B = next(iter(nobs.values())).shape[0]
        
        # process input
        obs_tokens = self.obs_encoder(nobs)
        # (B, N, n_emb)
        
        # empty data for action
        cond_data = torch.zeros(size=(B, self.action_horizon, self.action_dim), device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        
        # run sampling
        nsample = self.conditional_sample(
            condition_data=cond_data, 
            condition_mask=cond_mask,
            cond=obs_tokens,
            **self.kwargs)
        
        # unnormalize prediction
        assert nsample.shape == (B, self.action_horizon, self.action_dim)

        if self.use_ts_normalizer:
            # reshape back to original shape
            B, T, D = nsample.shape
            nsample = nsample.permute(0, 2, 1)  # (B, T, D) -> (B, D, T)
            if 'action' not in self.ts_normalizer_shape_cache.keys():
                self.ts_normalizer_shape_cache['action'] = (D, T)
            nsample = nsample.reshape(nsample.shape[:-2] + (1, T * D,))
        else:
            # unnormalize prediction
            assert nsample.shape == (B, self.action_horizon, self.action_dim)

        if (not self.use_embodiment_normalizer):
            action_pred = self.normalizer['action'].unnormalize(nsample)
        else:
            action_pred_robot = self.normalizer['action'].unnormalize(nsample)
            action_pred_human = self.normalizer_human['action'].unnormalize(nsample)
            if embodiment is not None:
                is_robot = embodiment.bool()
            else:
                is_robot = torch.ones(B, dtype=torch.bool, device=self.device)
            is_robot_expanded = is_robot.view(-1, *([1] * (action_pred_robot.dim() - 1)))
            action_pred = torch.where(is_robot_expanded, action_pred_robot, action_pred_human)
        
        if self.use_ts_normalizer:
            action_pred = action_pred.reshape(
                action_pred.shape[:-2] + self.ts_normalizer_shape_cache['action']
            )
            action_pred = action_pred.permute(0, 2, 1)  # (B, D, T) -> (B, T, D)

        result = {
            'action': action_pred,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer, normalizer_human: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
        if self.use_embodiment_normalizer:
            self.normalizer_human.load_state_dict(normalizer_human.state_dict())

    def compute_loss(self, batch):
        # normalize input


        if self.use_ts_normalizer: 
            for key in self.ts_normalizer_obs_key:
                T, D = batch['obs'][key].shape[-2:]
                if key not in self.ts_normalizer_shape_cache:
                    self.ts_normalizer_shape_cache[key] = (T, D)
                batch['obs'][key] = batch['obs'][key].reshape(batch['obs'][key].shape[:-2] + (1, T * D,))
            _, T, D = batch['action'].shape
            batch['action'] = batch['action'].permute(0, 2, 1)  # (B, T, D) -> (B, D, T)
            if 'action' not in self.ts_normalizer_shape_cache.keys():
                self.ts_normalizer_shape_cache['action'] = (D, T)
            batch['action'] = batch['action'].reshape(batch['action'].shape[:-2] + (1, T * D,))

        assert 'valid_mask' not in batch
        if (not self.use_embodiment_normalizer):
            nobs = self.normalizer.normalize(batch['obs'])
            nactions = self.normalizer['action'].normalize(batch['action'])
        else:
            nobs = self.normalizer.normalize(batch['obs'])
            nactions = self.normalizer['action'].normalize(batch['action'])
            nobs_human = self.normalizer_human.normalize(batch['obs'])
            nactions_human = self.normalizer_human['action'].normalize(batch['action'])
            is_robot = batch['embodiment'].bool()
            for key in nobs.keys():
                is_robot_expanded = is_robot.view(-1, *([1] * (nobs[key].dim() - 1)))
                nobs[key] = torch.where(is_robot_expanded, nobs[key], nobs_human[key])
            is_robot_expanded = is_robot.view(-1, *([1] * (nactions.dim() - 1)))
            nactions = torch.where(is_robot_expanded, nactions, nactions_human)

        if self.use_ts_normalizer:
            for key in self.ts_normalizer_obs_key:
                nobs[key] = nobs[key].reshape(
                    nobs[key].shape[:-2] + self.ts_normalizer_shape_cache[key]
                )
            nactions = nactions.reshape(
                nactions.shape[:-2] + self.ts_normalizer_shape_cache['action']
            )
            nactions = nactions.permute(0, 2, 1)  # (B, D, T) -> (B, T, D)

        trajectory = nactions
        
        # process input
        obs_tokens = self.obs_encoder(nobs)
        # (B, N, n_emb)
        
        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        # input perturbation by adding additonal noise to alleviate exposure bias
        # reference: https://github.com/forever208/DDPM-IP
        noise_new = noise + self.input_pertub * torch.randn(trajectory.shape, device=trajectory.device)

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (nactions.shape[0],), device=trajectory.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise_new, timesteps)
        
        # Predict the noise residual
        pred = self.model(
            noisy_trajectory,
            timesteps, 
            cond=obs_tokens
        )

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = (loss * batch['alpha'][..., None]).mean()

        return loss

    def forward(self, batch):
        return self.compute_loss(batch)