import math
import torch
import torch.nn as nn

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):                 
        device = x.device             # (B, ), the timestamps of diffusion (with batch)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]                    # (B, 1) * (1, diffusion_step_embed_dim // 2)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)    # (B, diffusion_step_embed_dim)
        return emb
