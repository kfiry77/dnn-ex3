import torch
import torch.nn as nn
import torch.nn.functional as F

class OneFcNet(nn.Module):
    def __init__(self):
        super(OneFcNet, self).__init__()
        self.fc = nn.Linear(32 * 32 * 3, 10)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = self.fc(x)
        return x
