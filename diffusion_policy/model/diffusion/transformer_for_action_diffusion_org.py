from typing import Union, Optional, Tuple
import logging
import torch
import torch.nn as nn
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from x_transformers import Encoder

logger = logging.getLogger(__name__)


class TransformerForActionDiffusion(ModuleAttrMixin):
    def __init__(self,
        input_dim: int,
        output_dim: int,
        diffusion_step_embed_dim: int,
        cond_dim: int,
        action_horizon: int,
        n_layer: int = 7,
        n_head: int = 8,
        n_emb: int = 768,                  # transformer_dim
        use_adaptive_layernorm: bool = True,
        use_adaptive_layerscale: bool = True
        ) -> None:
        super().__init__()
        
        # input embedding stem
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )
        self.pos_emb = nn.Parameter(torch.randn((1, action_horizon, n_emb)))

        self.dim_observation = diffusion_step_embed_dim + cond_dim

        self.diffusion_transformer = Encoder(
            dim=n_emb,
            depth=n_layer,
            heads=n_head,
            cross_attend=False,
            cross_attn_dim_context=n_emb,
            attn_dim_head=n_emb//n_head,
            dim_condition=self.dim_observation,
            use_adaptive_layernorm=use_adaptive_layernorm,
            use_adaptive_layerscale=use_adaptive_layerscale
        )

        # decoder head
        self.head = nn.Linear(n_emb, output_dim)
        self.action_horizon = action_horizon
        
        # init
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    
    def get_optim_groups(self, weight_decay: float=1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("_dummy_variable")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def configure_optimizers(self, 
            learning_rate: float=1e-4, 
            weight_decay: float=1e-3,
            betas: Tuple[float, float]=(0.9,0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def forward(self, 
        sample: torch.Tensor, 
        timestep: Union[torch.Tensor, float, int], 
        cond: Optional[torch.Tensor]=None, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        cond: (B,cond_dim)
        output: (B,T,input_dim)
        """
        
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps)  # (B, diffusion_step_embed_dim)
        
        # 2. process conditions
        cond_emb = torch.cat([time_emb, cond], dim=1)  # (B, diffusion_step_embed_dim + cond_dim)
        
        # 3. process input
        input_emb = self.input_emb(sample)
        t = input_emb.shape[1]
        pos_emb = self.pos_emb[:, :t, :] 
        input_emb = input_emb + pos_emb
        
        # 4. diffusion transformer
        x = self.diffusion_transformer(
            input_emb,
            condition=cond_emb
        )        
        x = self.head(x)
        # (B, T, n_out)
        return x
        
