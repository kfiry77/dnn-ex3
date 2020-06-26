import torch
import torch.nn as nn
import torch.nn.functional as F

class OneConvOneFcNet(nn.Module):
    def __init__(self):
        super(OneConvOneFcNet, self).__init__()
        self.conv = nn.Conv2d(3, 16, 6)
        self.pool = nn.MaxPool2d(4, 4)
        self.fc = nn.Linear(16 * 6 * 6, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = x.view(-1, 16 * 6 * 6)
        x = self.fc(x)
        return x
