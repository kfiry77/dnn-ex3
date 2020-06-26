import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoConvTwoFcNet(nn.Module):
    def __init__(self):
        super(TwoConvTwoFcNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def weights(self):
        return torch.cat((torch.flatten(self.conv1.weight.data),
                          torch.flatten(self.conv2.weight.data),
                          torch.flatten(self.fc1.weight),
                          torch.flatten(self.fc2.weight)
                         ))

