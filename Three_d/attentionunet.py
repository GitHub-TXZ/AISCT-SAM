import sys
if __name__ == '__main__':
    from common.unet_utils import inconv, down_block
    from common.utils import get_block, get_norm
    from common.conv_layers import BasicBlock, Bottleneck, ConvNormAct
else:
    from .common.unet_utils import inconv, down_block
    from .common.utils import get_block, get_norm
    from .common.conv_layers import BasicBlock, Bottleneck, ConvNormAct


import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, g_ch, l_ch, int_ch):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv3d(g_ch, int_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm3d(int_ch)
            )
        self.W_x = nn.Sequential(
            nn.Conv3d(l_ch, int_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm3d(int_ch)
            )
        self.psi = nn.Sequential(
            nn.Conv3d(int_ch, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm3d(1),
            nn.Sigmoid()
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: input low-res feature
        # x: high-res feature from encoder
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class attention_up_block(nn.Module):
    def __init__(self, in_ch, out_ch, num_block, block=BasicBlock, kernel_size=[3,3,3], up_scale=[2,2,2], norm=nn.BatchNorm3d):
        super().__init__()

        self.conv_ch = nn.Conv3d(in_ch, out_ch, kernel_size=1)

        self.up_scale = up_scale

        self.attn = AttentionBlock(in_ch, out_ch, out_ch//2)

        block_list = []
        block_list.append(block(in_ch+out_ch, out_ch, kernel_size=kernel_size, norm=norm))

        for i in range(num_block-1):
            block_list.append(block(out_ch, out_ch, kernel_size=kernel_size, norm=norm))

        self.conv = nn.Sequential(*block_list)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.shape[2:], mode='trilinear', align_corners=True)

        x2 = self.attn(x1, x2)


        out = torch.cat([x2, x1], dim=1)
        out = self.conv(out)

        return out




class AttentionUNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=32, scale=[[2, 2, 2], [2, 2, 2], 2, 2], kernel_size=[3, 3, 3, 3, 3], num_classes=2, block='SingleConv', pool=True, norm='bn'):
        super().__init__()

        num_block = 2
        block = get_block(block)
        norm = get_norm(norm)

        self.inc = inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)

        self.down1 = down_block(base_ch, 2 * base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0],
                                kernel_size=kernel_size[1], norm=norm)
        self.down2 = down_block(2 * base_ch, 4 * base_ch, num_block=num_block, block=block, pool=pool,
                                down_scale=scale[1], kernel_size=kernel_size[2], norm=norm)
        self.down3 = down_block(4 * base_ch, 8 * base_ch, num_block=num_block, block=block, pool=pool,
                                down_scale=scale[2], kernel_size=kernel_size[3], norm=norm)
        self.down4 = down_block(8 * base_ch, 10 * base_ch, num_block=num_block, block=block, pool=pool,
                                down_scale=scale[3], kernel_size=kernel_size[4], norm=norm)

        self.up1 = attention_up_block(10 * base_ch, 8 * base_ch, num_block=num_block, block=block, up_scale=scale[3],
                                      kernel_size=kernel_size[3], norm=norm)
        self.up2 = attention_up_block(8 * base_ch, 4 * base_ch, num_block=num_block, block=block, up_scale=scale[2],
                                      kernel_size=kernel_size[2], norm=norm)
        self.up3 = attention_up_block(4 * base_ch, 2 * base_ch, num_block=num_block, block=block, up_scale=scale[1],
                                      kernel_size=kernel_size[1], norm=norm)
        self.up4 = attention_up_block(2 * base_ch, base_ch, num_block=num_block, block=block, up_scale=scale[0],
                                      kernel_size=kernel_size[0], norm=norm)

        self.outc = nn.Conv3d(base_ch, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        out = self.up1(x5, x4)
        out = self.up2(out, x3)
        out = self.up3(out, x2)
        out = self.up4(out, x1)
        out = self.outc(out)

        return out

if __name__ == "__main__":
    # AttentionUNet 下采样倍数调整为1/4, 1/16, 1/16 参数量16M
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    from torchsummary import summary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # x = torch.randn(2, 1, 20, 320, 256)
    x = torch.randn(2, 1, 128, 128, 128)
    x = x.to(device)
    print("x size: {}".format(x.size()))

    # model = AttentionUNet(in_ch=1, base_ch=32, scale=[[2, 2, 2], [2, 2, 2], 2, 2], kernel_size=[3, 3, 3, 3, 3],
    #                       num_classes=2).to(device)
    model = AttentionUNet().to(device)
    out = model(x)
    print("out size: {}".format(out.size()))
    summary(model, input_size=(1, 128, 128, 128), device=device.type)
