import torch
import torch.nn as nn
import torch.nn.functional as F

class MEANet(nn.Module):
    def __init__(self, num_classes=10):
        super(MEANet, self).__init__()
        self.m1 = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.m2 = nn.Sequential(
            nn.Dropout(0.5),

            nn.Linear(3200, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, return_hidden=False, return_activation=False):
        if len(x.size()) == 3:
            x = x.unsqueeze(0)
        n = x.size(0)
        act1 = self.m1(x)
        x = F.adaptive_avg_pool2d(act1, (5, 5))
        hidden_feature = x.view(n, -1)
        x = self.m2(hidden_feature)
        if return_hidden:
            return x, hidden_feature
        elif return_activation:
            return x, act1, act1, act1
        else:
            return x
