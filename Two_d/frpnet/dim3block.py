import torch
from torch import nn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv7 = nn.Conv2d(2, 1, 7, padding=7 // 2, bias=False)
        self.conv5 = nn.Conv2d(2, 1, 5, padding=5 // 2, bias=False)
        self.conv3 = nn.Conv2d(2, 1, 3, padding=3 // 2, bias=False)
        self.conv1 = nn.Conv2d(2, 1, 1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        x_7 = self.conv7(y)
        x_5 = self.conv5(y)
        x_3 = self.conv3(y)
        x_1 = self.conv1(y)
        ax = self.sigmoid(x_7 + x_5 + x_3 + x_1)
        return x*ax


class d3block(nn.Module):

    def __init__(self, inchannel, outchannel, k_size=3):
        super(d3block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.yhw = SpatialAttention()
        # t = int(abs((math.log(channel, 2)+1)/2))  # gamma=2, b=1
        # k_size = t if t % 2 else t+1
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.inchannel = inchannel
        self.conv2 = nn.Conv2d(3 * inchannel, outchannel, kernel_size=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(inchannel, outchannel, kernel_size=1, padding=0, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        yc1 = self.avg_pool(x)
        yc2 = self.max_pool(x)  # b c h w

        x.permute(0, 2, 1, 3)  # b c h w 2 b h c w
        yh1 = self.avg_pool(x)
        yh2 = self.max_pool(x)

        x.permute(0, 3, 2, 1)  # b h c w 2 b w c h
        yw1 = self.avg_pool(x)
        yw2 = self.max_pool(x)

        yc = yc1 + yc2
        yc = self.conv(yc.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        yc = self.sigmoid(yc)

        yh = yh1 + yh2
        yh.permute(0, 2, 1, 3)
        yh = self.conv(yh.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        yh = self.sigmoid(yh)

        yw = yw1 + yw2
        yw.permute(0, 2, 3, 1)
        yw = self.conv(yw.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        yw = self.sigmoid(yw)

        x.permute(0, 2, 3, 1)  # b w c h 2 b c h w
        yhw = self.yhw(x)

        if self.inchannel >= 512:
            y = x * yc.expand_as(x) + x * yh.expand_as(x) + x * yw.expand_as(x)
            y = self.conv3(y)
        else:
            y = torch.cat([x * yc.expand_as(x), x * yh.expand_as(x), x * yw.expand_as(x)], dim=1)
            y = self.conv2(y)

        return x + y + yhw


class d1block(nn.Module):

    def __init__(self, inchannel, outchannel, k_size=3):
        super(d1block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # t = int(abs((math.log(channel, 2)+1)/2))  # gamma=2, b=1
        # k_size = t if t % 2 else t+1
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x,):

        y1 = self.avg_pool(x)
        y2 = self.max_pool(x)

        y = y1 + y2
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        y = self.sigmoid(y)

        return x * y.expand_as(x)

