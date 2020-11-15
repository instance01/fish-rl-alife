import torch
import torch.nn as nn


class SimpleDQN(torch.nn.Module):
    def __init__(self, channels, height, width, outputs):
        super(SimpleDQN, self).__init__()
        self.pre_head_dim = 16
        self.fc_net = nn.Sequential(
            nn.Linear(channels * height * width, 32),
            nn.ELU(),
            nn.Linear(32, self.pre_head_dim),
            nn.ELU(),
            nn.Linear(self.pre_head_dim, outputs))

    def forward(self, x):
        return self.fc_net(x.view(x.size(0), -1))


class ComplexDQN(torch.nn.Module):
    def __init__(self, channels, height, width, outputs):
        super(ComplexDQN, self).__init__()

        self.cnv_net = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ELU()
        )

        self.fc_net = nn.Sequential(
            nn.Linear(32 * height * width, 128),
            nn.ELU(),
            nn.Linear(128, outputs)
        )

    def forward(self, x):
        x = self.cnv_net(x)
        x = x.view(x.size(0), -1)
        return self.fc_net(x)
