import torch.nn as nn
import torch

class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # self.norm2 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)


    def forward(self, x):

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        # 上采样操作，通常使用反卷积或转置卷积
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        # 两个卷积层，可以加入批归一化和激活函数
        self.conv1 = nn.Conv2d(out_channels*2, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm3 = nn.InstanceNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x, encoder_features):
        x = self.upconv(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = torch.cat([x, encoder_features], dim=1) # 按通道拼接跳跃连接
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


# 构建UNet解码器
class Diff_Decoder(nn.Module):
    def __init__(self, diff_encoder_channels, diff_decoder_channels):
        super(Diff_Decoder, self).__init__()
        self.decoder_blocks = nn.ModuleList()

        # 反向循环遍历编码器通道数列表
        for in_channels, out_channels in zip(reversed(diff_encoder_channels),  diff_decoder_channels[1:]):
            self.decoder_blocks.append(DecoderBlock(in_channels, out_channels))

    def forward(self, encoder_features):
        x = encoder_features[-1]
        for decoder_block, encoder_feature in zip(self.decoder_blocks, reversed(encoder_features[:-1])):
            x = decoder_block(x, encoder_feature)  # 编解码块融合
        return x



class SPG(nn.Module):
    def __init__(self):
        super(SPG, self).__init__()
        # 定义两个共享权重的卷积分支
        self.branch1 = self._build_branch()
        self.branch2 = self._build_branch()
        self.diff_decoder = Diff_Decoder([16, 32, 64, 128, 256], [256, 128, 64, 32, 16])
        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)
        self.reset_parameters()

    def _build_branch(self):
        return nn.Sequential(
            ConvBlock2D(3, 16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock2D(16, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock2D(32, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock2D(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock2D(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def reset_parameters(self):
        for layer in self.branch1:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        for layer in self.branch2:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)


    def forward(self, x1, x2):
        y1 = x1
        outputs1 = []
        for layer in self.branch1:
            y1 = layer(y1)
            if not isinstance(layer, nn.MaxPool2d):
                outputs1.append(y1)
        outputs2 = []
        y2 = x2
        for layer in self.branch2:
            y2 = layer(y2)
            if not isinstance(layer, nn.MaxPool2d):
                outputs2.append(y2)

        diff_encoder_features = [i-j for i, j in zip(outputs1, outputs2)]
        diff = self.diff_decoder(diff_encoder_features)
        final_diff = torch.sigmoid(self.final_conv(diff))
        # bottleneck = torch.cat([outputs1[-1], outputs2[-1]], dim=1)
        bottleneck = torch.sigmoid(outputs1[-1] - outputs2[-1])
        return bottleneck, final_diff


if __name__ == '__main__':
    from einops import rearrange
    import numpy as np
    input_shape = (1, 3, 256, 256, 40) # (batch_size, channels, height, width, depth)
    input_data = torch.randn(input_shape)
    input_data = rearrange(input_data, 'b c h w d -> (b d) c h w')
    spg = SPG()
    model_parameters = filter(lambda p: p.requires_grad, spg.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"总参数数量：{params/1e6}M")
    # 获取每个层的参数数量和名称
    for name, param in spg.named_parameters():
        print(f"层名称: {name}, 参数形状: {param.shape}")

    total_params = sum(p.numel() for p in spg.parameters())
    print(f"Total parameters: {total_params / 1e6}M")
    trainable_params = sum(p.numel() for p in spg.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params / 1e6}M")
    bottleneck, final_diff= spg(input_data, input_data)
    pass