from .dim3block import *


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class hyencoder(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(hyencoder, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.Conv1 = DoubleConv(n_channels, 64)
        self.d3_1 = d3block(64, 64, k_size=3)
        self.maxpool = torch.nn.MaxPool2d(2)

        self.Conv2 = DoubleConv(64, 128)
        self.d3_2 = d3block(128, 128, k_size=5)

        self.Conv3 = DoubleConv(128, 256)
        self.d3_3 = d3block(256, 256, k_size=5)

        self.Conv4 = DoubleConv(256, 512)
        self.d3_4 = d3block(512, 512, k_size=5)

        self.Conv5 = DoubleConv(512, 1024)
        self.d3_5 = d3block(1024, 1024, k_size=5)
        self.drop = nn.Dropout(p=0.2)

    def forward(self, x):

        x1_1 = self.Conv1(x)
        x1_2 = self.d3_1(x1_1)
        x1_3 = self.maxpool(x1_2)

        x2_1 = self.Conv2(x1_3)
        x2_2 = self.d3_2(x2_1)
        x2_3 = self.maxpool(x2_2)

        x3_1 = self.Conv3(x2_3)
        x3_2 = self.d3_3(x3_1)
        x3_3 = self.maxpool(x3_2)

        x4_1 = self.Conv4(x3_3)
        x4_2 = self.d3_4(x4_1)
        x4_3 = self.maxpool(x4_2)
        x4_dp = self.drop(x4_3)

        x5_1 = self.Conv5(x4_dp)
        # x5_2 = self.d3_5(x5_1)
        # x5_dp = self.drop(x5_2)

        features = [x5_1, x4_2, x3_2, x2_2]
        return features
