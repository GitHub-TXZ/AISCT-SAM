import torch.nn as nn
import torch
if __name__ == '__main__':
    from Scconv import ScConv
else:
    from .Scconv import ScConv

class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = ScConv(out_channels)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        return out


def _build_branch(base_feature):
    base_feature = base_feature
    return nn.ModuleList([
        nn.Sequential(
            ConvBlock3D(1, base_feature),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        ),
        nn.Sequential(
            ConvBlock3D(base_feature, base_feature*2),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        ),
        # nn.Sequential(
        #     ConvBlock3D(base_feature*2, base_feature*4),
        #     nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # ),
        # nn.Sequential(
        #     ConvBlock3D(base_feature*4, base_feature*8),
        #     nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # )
    ])


class SPG(nn.Module):
    def __init__(self):
        super(SPG, self).__init__()
        self.branch1 = _build_branch(base_feature=32)
        self.out_conv = nn.Conv3d(64, 1, kernel_size=1)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.branch1:
            if isinstance(layer, nn.Conv3d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        nn.init.kaiming_normal_(self.out_conv.weight, mode='fan_out', nonlinearity='relu')
        if self.out_conv.bias is not None:
            nn.init.constant_(self.out_conv.bias, 0)    


    def forward(self, x1, x2):
        stage1 = self.branch1[0](x1)
        stage2 = self.branch1[1](stage1)
        
        stage1_sym = self.branch1[0](x2)
        stage2_sym = self.branch1[1](stage1_sym)
        sp = self.out_conv(stage2-stage2_sym)

        return sp



if __name__ == '__main__':
    from einops import rearrange
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "2"
    import numpy as np
    input_shape = (2, 1, 20, 256, 256)  # (batch_size, channels, height, width, depth)
    input_data = torch.randn(input_shape).cuda()
    spg = SPG().cuda()
    model_parameters = filter(lambda p: p.requires_grad, spg.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"总参数数量：{params / 1e6}M")
    # 获取每个层的参数数量和名称
    for name, param in spg.named_parameters():
        print(f"层名称: {name}, 参数形状: {param.shape}")

    total_params = sum(p.numel() for p in spg.parameters())
    print(f"Total parameters: {total_params / 1e6}M")
    trainable_params = sum(p.numel() for p in spg.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params / 1e6}M")
    semantic_prompts = spg(input_data,input_data)
    print(semantic_prompts.shape)
    pass
