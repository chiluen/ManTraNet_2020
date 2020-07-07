from sym_padding import *
import torch
import torch.nn.functional as F

class Featex_vgg16_base(nn.Module):
    def __init__(self, type=1):
        super(Featex_vgg16_base, self).__init__()
        base = 32
        self.type = type
        # block 1
        in_channels = 3
        out_channels = base # 32
        self.b1c1 = CombinedConv2D(in_channels, 32 if type in [0, 1] else 16) # relu
        self.b1c2 = Conv2DSymPadding(32 if type in [0, 1] else 16, out_channels, (3, 3)) # relu
        # block 2
        in_channels = out_channels
        out_channels = 2 * base # 64
        self.b2c1 = Conv2DSymPadding(in_channels, out_channels, (3, 3)) # relu
        self.b2c2 = Conv2DSymPadding(out_channels, out_channels, (3, 3)) # relu
        # block 3
        in_channels = out_channels
        out_channels = 4 * base # 128??
        self.b3c1 = Conv2DSymPadding(in_channels, out_channels, (3, 3)) # relu
        self.b3c2 = Conv2DSymPadding(out_channels, out_channels, (3, 3)) # relu
        self.b3c3 = Conv2DSymPadding(out_channels, out_channels, (3, 3)) # relu
        # block 4
        in_channels = out_channels
        out_channels = 8 * base # 256??
        self.b4c1 = Conv2DSymPadding(in_channels, out_channels, (3, 3)) # relu
        self.b4c2 = Conv2DSymPadding(out_channels, out_channels, (3, 3)) # relu
        self.b4c3 = Conv2DSymPadding(out_channels, out_channels, (3, 3)) # relu
        # block 5/bottle-neck
        self.b5c1 = Conv2DSymPadding(out_channels, out_channels, (3, 3)) # relu
        self.b5c2 = Conv2DSymPadding(out_channels, out_channels, (3, 3)) # relu
        self.transform = Conv2DSymPadding(out_channels, out_channels, (3, 3)) # tanh if type >= 1 else None
        # l2_norm

    def forward(self, x):
        # block 1
        x = F.relu(self.b1c1(x))
        x = F.relu(self.b1c2(x))
        # block 2
        x = F.relu(self.b2c1(x))
        x = F.relu(self.b2c2(x))
        # block 3
        x = F.relu(self.b3c1(x))
        x = F.relu(self.b3c2(x))
        x = F.relu(self.b3c3(x))
        # block 4
        x = F.relu(self.b4c1(x))
        x = F.relu(self.b4c2(x))
        x = F.relu(self.b4c3(x))
        # block 5
        x = F.relu(self.b5c1(x))
        x = F.relu(self.b5c2(x))
        x = self.transform(x) if self.type >= 1 else torch.tanh(self.transform(x))
        # l2 normalization
        x = F.normalize(x, p=2, dim=1)
        return x