import torch
from torch import nn

# MMD-Loss Reference: https://github.com/yiftachbeer/mmd_loss_pytorch/blob/master/mmd_loss.py

class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        if self.bandwidth_multipliers.device != L2_distances.device:
            self.bandwidth_multipliers = self.bandwidth_multipliers.to(L2_distances.device)
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


class MMDLoss(nn.Module):

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY


def compute_mmd_loss(mmd_loss_block, features, label, device):
    real_mask = (label == 1).view(-1)
    fake_mask = (label == 0).view(-1)
    
    real_samples = features[real_mask]
    fake_samples = features[fake_mask]
    
    if real_samples.size(0) == 0 or fake_samples.size(0) == 0:
        return torch.tensor(0.0, device=device)
    
    mmd_loss = mmd_loss_block(real_samples, fake_samples)
    return mmd_loss


def compute_gradient_penalty(judger_model, features, label, device):

    real_mask = (label == 1).view(-1)
    fake_mask = (label == 0).view(-1)
    real_samples = features[real_mask]
    fake_samples = features[fake_mask]
    
    if real_samples.size(0) == 0 or fake_samples.size(0) == 0:
        return torch.tensor(0.0, device=device)
    use_batch_size = min(real_samples.size(0), fake_samples.size(0))
    alpha = torch.rand(use_batch_size, 1).to(device)
    interpolates = alpha * real_samples[:use_batch_size] + (1 - alpha) * fake_samples[:use_batch_size]
    interpolates = interpolates.requires_grad_(True)

    d_interpolates = judger_model(interpolates)
    fake = torch.ones(d_interpolates.size()).to(device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty
