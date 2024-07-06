import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RewardNet(nn.Module):
    def __init__(self, feat_dim,hidden=50,hidden2 = 10):
        super(RewardNet, self).__init__()
        self.fc1 = nn.Linear(feat_dim, hidden)
        self.fc2 = nn.Linear(hidden,hidden2)
        self.fc3 = nn.Linear(hidden2, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
class RewardNet1D(nn.Module):
    def __init__(self, feat_dim,hidden=50):
        super(RewardNet1D, self).__init__()
        self.fc1 = nn.Linear(feat_dim, hidden)
        # self.fc2 = nn.Linear(hidden,hidden2)
        self.fc3 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x