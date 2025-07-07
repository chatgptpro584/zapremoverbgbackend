# models/u2net.py

import torch
import torch.nn as nn

class U2NET(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(U2NET, self).__init__()
        # Simplified structure here – include full U2NET class from the official repo
        self.stem = nn.Conv2d(in_ch, 64, 3, padding=1)
        self.out = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.out(x)
        return x, x, x, x, x, x, x  # placeholder for 7-stage U²-Net outputs
