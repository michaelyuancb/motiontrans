import torch
import torch.nn as nn
from diffusion_policy.model.talk2dino.src.model import ProjectionLayer

class Talk2DINO_Wrapper(nn.Module):
    def __init__(self, config_path, weights_path, out_dim, frozen=False):
        super(Talk2DINO_Wrapper, self).__init__()
        talk2dino = ProjectionLayer.from_config(config_path)
        talk2dino.load_state_dict(torch.load(weights_path, map_location='cpu'))
        self.talk2dino = talk2dino
        self.frozen = frozen
        self.out_dim = out_dim
        if self.frozen:
            for param in self.talk2dino.parameters():
                param.requires_grad = False
        else:
            for param in self.talk2dino.parameters():
                param.requires_grad = True

        self.proj_out = nn.Linear(768, out_dim)

    def forward(self, text_features):
        feat = self.talk2dino.project_clip_txt(text_features)
        feat = self.proj_out(feat)  # Apply mean pooling
        return feat