import torch
import torch.nn as nn

class EmbodimentAdversarialNetworkNN(nn.Module):

    def __init__(self, obs_feature_dim, hidden_dim=[512, 128, 32], activate="relu"):

        super(EmbodimentAdversarialNetworkNN, self).__init__()
        self.obs_feature_dim = obs_feature_dim
        self.hidden_dim = hidden_dim
        if activate == "relu":
            self.activate = nn.ReLU()
        elif activate == "tanh":
            self.activate = nn.Tanh()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(obs_feature_dim, hidden_dim[0]))
        for i in range(len(hidden_dim) - 1):
            self.layers.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            self.layers.append(self.activate)
        self.layers.append(nn.Linear(hidden_dim[-1], 1))
        self.layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*self.layers) 

    def forward(self, x):
        x = self.layers(x)
        return x

