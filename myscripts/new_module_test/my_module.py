import os
import torch
from torch import nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_
import math



import torch
import torch.nn as nn
import torch.nn.functional as F


class MSCA(nn.Module):
    def __init__(self, channels):
        super(MSCA, self).__init__()
        self.d = channels // 8
        self.conv_5x5 = nn.Conv2d(channels, self.d, kernel_size=5, padding=2, groups=self.d)
        self.conv_1x7 = nn.Conv2d(self.d, self.d, kernel_size=(1, 7), padding=(0, 3), groups=self.d)
        self.conv_7x1 = nn.Conv2d(self.d, self.d, kernel_size=(7, 1), padding=(3, 0), groups=self.d)
        self.conv_1x11 = nn.Conv2d(self.d, self.d, kernel_size=(1, 11), padding=(0, 5), groups=self.d)
        self.conv_11x1 = nn.Conv2d(self.d, self.d, kernel_size=(11, 1), padding=(5, 0), groups=self.d)
        self.conv_1x21 = nn.Conv2d(self.d, self.d, kernel_size=(1, 21), padding=(0, 10), groups=self.d)
        self.conv_21x1 = nn.Conv2d(self.d, self.d, kernel_size=(21, 1), padding=(10, 0), groups=self.d)
        self.channel_mixing = nn.Conv2d(self.d * 6, channels, kernel_size=1)
        self.conv_attention = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        # Multi-scale feature
        out_5x5 = self.conv_5x5(x)

        # Generate multi-scale features using different kernel sizes
        out_1x7 = self.conv_1x7(out_5x5)
        out_7x1 = self.conv_7x1(out_5x5)
        out_1x11 = self.conv_1x11(out_5x5)
        out_11x1 = self.conv_11x1(out_5x5)
        out_1x21 = self.conv_1x21(out_5x5)
        out_21x1 = self.conv_21x1(out_5x5)

        # Concatenate the multi-scale features
        multi_scale_features = torch.cat([out_1x7, out_7x1, out_1x11, out_11x1, out_1x21, out_21x1], dim=1)

        # Channel mixing
        channel_mixed = self.channel_mixing(multi_scale_features)

        # Convolutional attention
        conv_attention = self.conv_attention(x)
        conv_attention = F.sigmoid(conv_attention)

        # Apply attention to the input
        out = x * conv_attention

        # Element-wise sum of channel mixed features and the attention-applied features
        out = out + channel_mixed

        return out

if __name__ == '__main__':
    # Test the MSCA module
    channels = 64  # example number of channels
    msca = MSCA(channels=channels)
    # Dummy input tensor for testing
    input_tensor = torch.randn(1, channels, 64, 64)
    # Forward pass through MSCA module
    output = msca(input_tensor)
    print(output.shape)  # Expected output shape: (1, channels, 64, 64)






