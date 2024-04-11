import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RewardNet(nn.Module):
    def __init__(self, feat_dim):
        super(RewardNet, self).__init__()
        self.fc1 = nn.Linear(feat_dim, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x