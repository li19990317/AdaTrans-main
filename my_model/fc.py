import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FCN, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.fc1 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 1)

    def forward(self, x):
        x = self.fc(x)
        x = self.fc1(x)
        out = self.out(x)

        return out

