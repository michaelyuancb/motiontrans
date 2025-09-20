import torch
import torch.nn as nn
import einops
from diffusion_policy.model.diffusion.conv1d_components import (Downsample1d, Upsample1d, Conv1dBlock)
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalResidualBlock1D

class EmbodimentAdversarialNetwork(nn.Module):

    def __init__(self, 
                 obs_feature_dim, 
                 down_dims=[256,512,1024],
                 kernel_size=5,
                 n_groups=8,
                 wasserstein=False):

        super(EmbodimentAdversarialNetwork, self).__init__()

        input_dim = 1
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        cond_dim = obs_feature_dim

        self.input_dim = input_dim 
        self.obs_feature_dim = obs_feature_dim

        # reference: https://cdn.aaai.org/ojs/11784/11784-13-15312-1-2-20201228.pdf
        self.wasserstein = wasserstein

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        self.in_out = in_out

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=True
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=True
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=True),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=True),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=True),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=True)
            ]))
        
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv
        if self.wasserstein is False:
            self.sigmoid = nn.Sigmoid()


    def forward(self, global_feature):

        B, F = global_feature.shape
        x = torch.zeros(size=(B, 1, 1), device=global_feature.device, dtype=global_feature.dtype)
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)

        x = self.final_conv(x)
        if self.wasserstein:
            x = x.reshape(B, 1)
        else:
            x = self.sigmoid(x).reshape(B, 1)
        return x


if __name__ == "__main__":
    network = EmbodimentAdversarialNetwork(
        obs_feature_dim=768
    )
    feat = torch.randn((64, 768))
    network = network.to('cuda')
    feat = feat.to('cuda')
    pred = network(feat)
    print(pred.shape)

# python -m diffusion_policy.policy.embodiment_adversarial_network