import torch
import torch.nn as nn
import torch.nn.functional as F

class OneConvOneFcNet(nn.Module):
    def __init__(self):
        super(OneConvOneFcNet, self).__init__()
#        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 16, 5)
        self.fc = nn.Linear(16 * 5 * 5, 10)

    def forward(self, x):
#        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.fc(x)
        return x
