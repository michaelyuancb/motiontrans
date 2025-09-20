from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.timm_obs_encoder import TimmObsEncoder
from common.pytorch_util import dict_apply


class DiffusionUnetHRA3TimmPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: TimmObsEncoder,
            adversarial_type="rgb_low_dim",
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            input_pertub=0.1,
            inpaint_fixed_action_prefix=False,
            train_diffusion_n_samples=1,
            use_embodiment_normalizer=False,
            # parameters passed to step
            **kwargs
        ):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        action_horizon = shape_meta['action']['horizon']
        # get feature dim
        obs_feature_dim = np.prod(obs_encoder.output_shape())
        self.use_embodiment_normalizer = use_embodiment_normalizer
        self.adversarial_type = adversarial_type
        obs_adv_feature_dim = np.prod(obs_encoder.output_shape_feat_type(return_feat=self.adversarial_type))
        self.obs_adv_feature_dim = obs_adv_feature_dim

        # create diffusion model
        assert obs_as_global_cond
        input_dim = action_dim
        global_cond_dim = obs_feature_dim

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.normalizer_human = LinearNormalizer()
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon # used for training
        self.obs_as_global_cond = obs_as_global_cond
        self.input_pertub = input_pertub
        self.inpaint_fixed_action_prefix = inpaint_fixed_action_prefix
        self.train_diffusion_n_samples = int(train_diffusion_n_samples)
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ========= inference  ============
    def conditional_sample(self, 
            condition_data,
            condition_mask,
            local_cond=None,
            global_cond=None,
            generator=None,
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
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def update_embedding_with_idx(self, index, new_num_embeddings):
        self.obs_encoder.update_embedding_with_idx(index, new_num_embeddings)

    def update_embedding(self, key, new_num_embeddings):
        self.obs_encoder.update_embedding(key, new_num_embeddings)
        

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], fixed_action_prefix: torch.Tensor=None, embodiment: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        fixed_action_prefix: unnormalized action prefix
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]

        # condition through global feature
        global_cond = self.obs_encoder(nobs)

        # empty data for action
        cond_data = torch.zeros(size=(B, self.action_horizon, self.action_dim), device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        if fixed_action_prefix is not None and self.inpaint_fixed_action_prefix:
            n_fixed_steps = fixed_action_prefix.shape[1]
            cond_data[:, :n_fixed_steps] = fixed_action_prefix
            cond_mask[:, :n_fixed_steps] = True
            cond_data = self.normalizer['action'].normalize(cond_data)

        # run sampling
        nsample = self.conditional_sample(
            condition_data=cond_data, 
            condition_mask=cond_mask,
            local_cond=None,
            global_cond=global_cond,
            **self.kwargs)
        
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

    def compute_loss(self, batch, debug_verify_action=False):
        # normalize input
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
        
        assert self.obs_as_global_cond
        global_cond, adv_feat = self.obs_encoder(nobs, return_feat=self.adversarial_type)

        # train on multiple diffusion samples per obs
        if self.train_diffusion_n_samples != 1:
            raise NotImplementedError("For HRA3 Algorithm, policy.train_diffusion_n_samples must be set to 1")

        trajectory = nactions
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
            local_cond=None,
            global_cond=global_cond
        )

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        elif pred_type == 'v_prediction':
            # v = alpha_t * noise - sqrt(1 - alpha_t) * x0
            alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(trajectory.device)  # [T]
            alphas_t = alphas_cumprod[timesteps].view(-1, *([1] * (noise.ndim - 1)))
            sqrt_alphas = alphas_t.sqrt()
            sqrt_one_minus_alphas = (1.0 - alphas_t).sqrt()
            target = sqrt_alphas * noise - sqrt_one_minus_alphas * trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")
        
        if debug_verify_action:
            import pdb; pdb.set_trace()
            pred_action = self.predict_action(batch['obs'])['action_pred']
            pred_act = self.predict_action(batch['obs'], None)['action_pred']

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        # NOTE: use alpha-weight to simulate weighted sampling
        loss = (loss * batch['alpha'][..., None]).mean()

        return loss, adv_feat


    def forward(self, batch, debug_verify_action=False):
        return self.compute_loss(batch, debug_verify_action)