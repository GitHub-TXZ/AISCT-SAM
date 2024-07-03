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


class Fourierblock(nn.Module):
    def __init__(self, dim, d=20, h=320, w=256):
        super().__init__()
        # 对于 rfftn，最后一个维度应该是 W//2 + 1，这里使用了 W//2 是因为 rfftn 的输出在最后一个维度上是压缩的
        self.complex_weight = nn.Parameter(torch.randn(d, h, w//2 + 1, dim, 2, dtype=torch.float32))
        self.bias = nn.Parameter(torch.randn(d, h, w//2+1, dim, dtype=torch.float32))
        self.w = w
        self.h = h
        self.d = d

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = torch.fft.rfftn(x, dim=(1, 2, 3), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight + self.bias
        x = torch.fft.irfftn(x, s=(D, H, W), dim=(1, 2, 3), norm='ortho')
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x



class MSCA(nn.Module):
    def __init__(self, inchannels=32, d=0, h=0, w=0, encoder_stage=1):
        super(MSCA, self).__init__()
        self.encoder_stage = encoder_stage
        # 定义不同大小的卷积核
        self.conv_3x3x3 = nn.Conv3d(inchannels, inchannels, kernel_size=(3, 3, 3), padding=(1, 1, 1), groups=inchannels)

        self.conv_1x5x5 = nn.Conv3d(inchannels, inchannels, kernel_size=(1, 5, 5), padding=(0, 2, 2), groups=inchannels)
        self.conv_5x1x1 = nn.Conv3d(inchannels, inchannels, kernel_size=(5, 1, 1), padding=(2, 0, 0), groups=inchannels)

        self.conv_1x7x7 = nn.Conv3d(inchannels, inchannels, kernel_size=(1, 7, 7), padding=(0, 3, 3), groups=inchannels)
        self.conv_7x1x1 = nn.Conv3d(inchannels, inchannels, kernel_size=(7, 1, 1), padding=(3, 0, 0), groups=inchannels)

        self.conv_1x11x11 = nn.Conv3d(inchannels, inchannels, kernel_size=(1, 11, 11), padding=(0, 5, 5), groups=inchannels)
        self.conv_11x1x1 = nn.Conv3d(inchannels, inchannels, kernel_size=(11, 1, 1), padding=(5, 0, 0), groups=inchannels)

        # 定义 1x1x1 卷积
        self.conv_1x1x1 = nn.Conv3d(inchannels, inchannels, kernel_size=(1, 1, 1))

        # 定义可选的傅里叶变换层
        if self.encoder_stage in [0,1,2,3]:
            pass
        else:
            self.fourier = Fourierblock(dim=inchannels,d=d,h=h,w=w)



    def forward(self, x):
        residue = x
        # 应用不同的卷积核
        conv_in = self.conv_3x3x3(x)
        conv_br1 = self.conv_5x1x1(self.conv_1x5x5(conv_in))
        conv_br2 = self.conv_7x1x1(self.conv_1x7x7(conv_in))
        conv_br3 = self.conv_11x1x1(self.conv_1x11x11(conv_in))
        if self.encoder_stage in [0,1,2,3]:
            combined = conv_in+conv_br1+conv_br2+conv_br3
            attn = self.conv_1x1x1(combined)
        else:
            conv_fourier = self.fourier(x)
            combined = conv_in + conv_br1 + conv_br2 + conv_br3 + conv_fourier
            attn = self.conv_1x1x1(combined)
        out = attn * x
        return out+x





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


class Decoder():
    def __init__(self, do_ds:bool=True):
        self.deep_supervision = do_ds


class I2PC_Net(nn.Module):
    def __init__(self, in_ch=1, base_ch=32, scale=[[1, 2, 2], [1, 2, 2],[1, 2, 2] ,2, 2,[1,2,2]], kernel_size=[[1,3,3], [1,3,3], 3, 3, 3, 3], num_classes=3, block='SingleConv', pool=False, norm='bn'):
        super(I2PC_Net, self).__init__()

        num_block = 2
        block = get_block(block)
        norm = get_norm(norm)

        self.inc = inconv(in_ch, base_ch, block=block, kernel_size=[1,3,3], norm=norm)
        self.msca0 = MSCA(base_ch,d=20,h=320,w=256,encoder_stage=0)
        self.down1 = down_block(base_ch, 2 * base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0],
                                kernel_size=kernel_size[0], norm=norm)
        self.msca1 = MSCA(base_ch*2,d=20,h=160,w=128,encoder_stage=1)
        self.down2 = down_block(2 * base_ch, 4 * base_ch, num_block=num_block, block=block, pool=pool,
                                down_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        self.msca2 = MSCA(base_ch * 4,d=20,h=80,w=64,encoder_stage=2)
        self.down3 = down_block(4 * base_ch, 8 * base_ch, num_block=num_block, block=block, pool=pool,
                                down_scale=scale[2], kernel_size=kernel_size[2], norm=norm)
        self.msca3 = MSCA(base_ch * 8,d=20,h=40,w=32,encoder_stage=3)
        self.down4 = down_block(8 * base_ch, 10 * base_ch, num_block=num_block, block=block, pool=pool,
                                down_scale=scale[3], kernel_size=kernel_size[3], norm=norm)
        self.msca4 = MSCA(base_ch * 10,d=10, h=20, w=16, encoder_stage=4)
        self.down5 = down_block(10 * base_ch, 10 * base_ch, num_block=num_block, block=block, pool=pool,
                                down_scale=scale[4], kernel_size=kernel_size[4], norm=norm)
        self.msca5 = MSCA(base_ch * 10, d=5, h=10, w=8, encoder_stage=5)
        self.down6 = down_block(10 * base_ch, 10 * base_ch, num_block=num_block, block=block, pool=pool,
                                down_scale=scale[5], kernel_size=kernel_size[5], norm=norm)
        self.msca6 = MSCA(base_ch * 10, d=5, h=5, w=4, encoder_stage=6)

        self.decoder = Decoder() #兼容nnUNet的深度监督
        self.up1 = attention_up_block(10 * base_ch, 10 * base_ch, num_block=num_block, block=block, up_scale=scale[5],
                                      kernel_size=kernel_size[5], norm=norm)
        self.up2 = attention_up_block(10 * base_ch, 10 * base_ch, num_block=num_block, block=block, up_scale=scale[4],
                                      kernel_size=kernel_size[4], norm=norm)
        self.up3 = attention_up_block(10 * base_ch, 8 * base_ch, num_block=num_block, block=block, up_scale=scale[3],
                                      kernel_size=kernel_size[3], norm=norm)
        self.up4 = attention_up_block(8 * base_ch, 4 * base_ch, num_block=num_block, block=block, up_scale=scale[2],
                                      kernel_size=kernel_size[2], norm=norm)
        self.up5 = attention_up_block(4 * base_ch, 2 * base_ch, num_block=num_block, block=block, up_scale=scale[1],
                                      kernel_size=kernel_size[1], norm=norm)
        self.up6 = attention_up_block(2 * base_ch, base_ch, num_block=num_block, block=block, up_scale=scale[0],
                                      kernel_size=kernel_size[0], norm=norm)

        self.outc = nn.Conv3d(base_ch, num_classes, kernel_size=1)
        self.seg_layers = nn.ModuleList([])
        self.seg_layers.append(nn.Conv3d(320, 2, kernel_size=1,stride=1,padding=0))
        self.seg_layers.append(nn.Conv3d(320, 2, kernel_size=1, stride=1, padding=0))
        self.seg_layers.append(nn.Conv3d(256, 2, kernel_size=1, stride=1, padding=0))
        self.seg_layers.append(nn.Conv3d(128, 3, kernel_size=1, stride=1, padding=0))
        self.seg_layers.append(nn.Conv3d(64, 3, kernel_size=1, stride=1, padding=0))

        self.asym_enhance1 = nn.Conv3d(256*2, 256, kernel_size=1)
        self.asym_enhance2 = nn.Conv3d(320*2, 320, kernel_size=1)
        self.asym_enhance3 = nn.Conv3d(320*2, 320, kernel_size=1)
        self.asym_enhance4 = nn.Conv3d(320*2, 320, kernel_size=1)




    def forward(self, x):
        x1 = self.msca0(self.inc(x))
        x2 = self.msca1(self.down1(x1))
        x3 = self.msca2(self.down2(x2))

        x4 = self.msca3(self.down3(x3))
        x4_flip = x4.flip(-1)      #flip不是原地操作
        x4 = self.asym_enhance1(torch.cat([x4-x4_flip, x4],dim=1)) + x4

        x5 = self.msca4(self.down4(x4))
        x5_flip = x5.flip(-1)
        x5 = self.asym_enhance2(torch.cat([x5 - x5_flip, x5],dim=1)) + x5

        x6 = self.msca5(self.down5(x5))
        x6_flip = x6.flip(-1)
        x6 = self.asym_enhance3(torch.cat([x6 - x6_flip, x6],dim=1)) + x6

        x7 = self.msca6(self.down6(x6))
        x7_flip = x7.flip(-1)  # flip不是原地操作
        x7 = self.asym_enhance3(torch.cat([x7 - x7_flip, x7],dim=1)) + x7


        out1 = self.up1(x7, x6)
        out2 = self.up2(out1, x5)
        out3 = self.up3(out2, x4)
        out4 = self.up4(out3, x3)
        out5 = self.up5(out4, x2)
        out6 = self.up6(out5, x1)

        ds1 = self.seg_layers[0](out1)
        ds2 = self.seg_layers[1](out2)
        ds3 = self.seg_layers[2](out3)
        ds4 = self.seg_layers[3](out4)
        ds5 = self.seg_layers[4](out5)
        out = self.outc(out6)
        if self.decoder.deep_supervision == True:
            return out,ds5,ds4,ds3,ds2,ds1
        else:
            return out

if __name__ == "__main__":
    # AttentionUNet 下采样倍数调整为1/4, 1/16, 1/16 参数量16M
    import os
    from thop import profile
    from thop import clever_format
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    from torchsummary import summary
    from torch import autocast


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input = torch.randn(2, 1, 20, 320, 256)
    input = input.to(device)
    model = I2PC_Net(in_ch=1, base_ch=32, num_classes=3).to(device)
    criterion = nn.MSELoss()
    # with autocast(device.type, enabled=True): cuFFT only supports dimensions whose sizes are powers of two when computing in half precision
    out = model(input)[0]
    loss = criterion(out, torch.ones_like(out).to(device))  # 假设我们的目标是预测一个全1的输出
    loss.backward()
    print("input size: {}".format(input.size()))
    print("out size: {}".format(out.size()))


    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")  # 格式化显示FLOPs和参数量
    print(f"FLOPs: {flops}, Params: {params}")

    # summary(model, input_size=(1, 128, 128, 128), device=device.type)