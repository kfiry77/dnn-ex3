import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoFcNet(nn.Module):
    def __init__(self):
        super(TwoFcNet, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
