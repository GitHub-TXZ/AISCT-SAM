""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
from einops import rearrange

from .unet_parts import *
from .lambda_networks import *


class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down( 64, 128)
        self.down2 = Down( 128, 256)
        self.down3 = Down( 256, 512)
        self.pool = nn.MaxPool2d(2)
        factor = 2
        self.down4 = Down( 512, 1024 // factor)
        # self.up1 = Up( 1024, 512 // factor, False, True)
        # self.up2 = Up( 512, 256 // factor, False, True)
        # self.up3 = Up( 256, 128 // factor, False, True)
        # self.up4 = Up( 128, 64, False, True)
        self.up1 = Up(1024, 512 // factor, True, False)
        self.up2 = Up(512, 256 // factor, True, False)
        self.up3 = Up(256, 128 // factor, True, False)
        self.up4 = Up(128, 64, True, False)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
