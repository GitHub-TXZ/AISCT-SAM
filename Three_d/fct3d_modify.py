from __future__ import annotations

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from myscripts.DynamicLargeKernelAttn.DLKA import TransformerBlock_3D_single_deform_LKA
from monai.networks.blocks.convolutions import Convolution
from monai.networks.blocks.segresnet_block import ResBlock, get_conv_layer, get_upsample_layer
from monai.networks.layers.factories import Dropout
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import UpsampleMode

from mamba_ssm import Mamba

class MambaLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
            d_model=input_dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm) + self.skip_scale * x_flat
        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out

def get_mamba_layer(
        spatial_dims: int, in_channels: int, out_channels: int, stride: int = 1
):
    mamba_layer = MambaLayer(input_dim=in_channels, output_dim=out_channels)
    if stride != 1:
        if spatial_dims == 2:
            return nn.Sequential(mamba_layer, nn.MaxPool2d(kernel_size=stride, stride=stride))
        if spatial_dims == 3:
            return nn.Sequential(mamba_layer, nn.MaxPool3d(kernel_size=stride, stride=stride))
    return mamba_layer


class ResMambaBlock(nn.Module):

    def __init__(
            self,
            spatial_dims: int = 3,
            in_channels: int = 8,
            norm: tuple | str = ("GROUP", {"num_groups": 8}),
            kernel_size: int = 3,
            act: tuple | str = ("RELU", {"inplace": True}),
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2 or 3.
            in_channels: number of input channels.
            norm: feature normalization type and arguments.
            kernel_size: convolution kernel size, the value should be an odd number. Defaults to 3.
            act: activation type and arguments. Defaults to ``RELU``.
        """

        super().__init__()

        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size should be an odd number.")

        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act = get_act_layer(act)
        self.conv1 = get_mamba_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels
        )
        self.conv2 = get_mamba_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels
        )

    def forward(self, x):
        identity = x

        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        x += identity

        return x

class Transformer(nn.Module):

    def __init__(self,
                 # in_channels,
                 out_channels,
                 num_heads,
                 dpr,
                 proj_drop=0.0,
                 attention_bias=True,
                 padding_q="same",
                 padding_kv="same",
                 stride_kv=1,
                 stride_q=1):
        super().__init__()

        self.RVM = ResMambaBlock(in_channels=out_channels)
        self.conv1 = nn.Conv3d(out_channels, out_channels, 3, 1, padding="same")
        self.layernorm = nn.LayerNorm(self.conv1.out_channels, eps=1e-5)
        self.dlka = TransformerBlock_3D_single_deform_LKA(input_size=out_channels, hidden_size=out_channels)



    def forward(self, x):
        # x1 = self.RVM(x)
        x1 = x
        x1 = self.conv1(x1)
        x2 = torch.add(x1, x)
        x3 = x2.permute(0, 2, 3, 4, 1)
        x3 = self.layernorm(x3)
        x3 = x3.permute(0, 4, 1, 2, 3)
        x3 = self.dlka(x3)
        # x3 = torch.add(x2, x3)
        return x3

class Block_encoder_bottleneck(nn.Module):
    def __init__(self, blk, in_channels, out_channels, att_heads, dpr):
        super().__init__()
        self.blk = blk
        if ((self.blk == "first") or (self.blk == "bottleneck")):
            self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
            self.conv1 = nn.Conv3d(in_channels, out_channels, 3, 1, padding="same")
            self.conv2 = nn.Conv3d(out_channels, out_channels, 3, 1, padding="same")
            self.trans = Transformer(out_channels, att_heads, dpr)
        elif ((self.blk == "second") or (self.blk == "third") or (self.blk == "fourth")):
            self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
            self.conv1 = nn.Conv3d(1, in_channels, 3, 1, padding="same")
            self.conv2 = nn.Conv3d(out_channels, out_channels, 3, 1, padding="same")
            self.conv3 = nn.Conv3d(out_channels, out_channels, 3, 1, padding="same")
            self.trans = Transformer(out_channels, att_heads, dpr)

    def forward(self, x, scale_img="none"):
        if ((self.blk == "first") or (self.blk == "bottleneck")):
            x1 = x.permute(0, 2, 3, 4, 1)
            x1 = self.layernorm(x1)
            x1 = x1.permute(0, 4, 1, 2, 3)
            x1 = F.relu(self.conv1(x1))
            x1 = F.relu(self.conv2(x1))
            # x1 = F.dropout(x1, 0.3)
            x1 = F.max_pool3d(x1, (2, 2, 2))
            out = self.trans(x1)
            # without skip
        elif ((self.blk == "second") or (self.blk == "third") or (self.blk == "fourth")):
            x1 = x.permute(0, 2, 3, 4, 1)
            x1 = self.layernorm(x1)
            x1 = x1.permute(0, 4, 1, 2, 3)
            x1 = torch.cat((F.relu(self.conv1(scale_img)), x1), axis=1)
            x1 = F.relu(self.conv2(x1))
            x1 = F.relu(self.conv3(x1))
            x1 = F.dropout(x1, 0.3)
            x1 = F.max_pool3d(x1, (2, 2, 2))
            out = self.trans(x1)
            # with skip
        return out


class Block_decoder(nn.Module):
    def __init__(self, in_channels, out_channels, att_heads, dpr):
        super().__init__()
        self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, 1, padding="same")
        self.conv2 = nn.Conv3d(out_channels * 2, out_channels, 3, 1, padding="same")
        self.conv3 = nn.Conv3d(out_channels, out_channels, 3, 1, padding="same")
        self.trans = Transformer(out_channels, att_heads, dpr)

    def forward(self, x, skip):
        x1 = x.permute(0, 2, 3, 4, 1)
        x1 = self.layernorm(x1)
        x1 = x1.permute(0, 4, 1, 2, 3)
        x1 = self.upsample(x1)
        x1 = F.relu(self.conv1(x1))
        x1 = torch.cat((skip, x1), axis=1)
        x1 = F.relu(self.conv2(x1))
        x1 = F.relu(self.conv3(x1))
        # x1 = F.dropout(x1, 0.3)
        out = self.trans(x1)
        return out


class DS_out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
        self.conv1 = nn.Conv3d(in_channels, in_channels, 3, 1, padding="same")
        self.conv2 = nn.Conv3d(in_channels, in_channels, 3, 1, padding="same")
        self.conv3 = nn.Conv3d(in_channels, out_channels, 1, 1, padding="same")

    def forward(self, x):
        x1 = self.upsample(x)
        x1 = x1.permute(0, 2, 3, 4, 1)
        x1 = self.layernorm(x1)
        x1 = x1.permute(0, 4, 1, 2, 3)
        x1 = F.relu(self.conv1(x1))
        x1 = F.relu(self.conv2(x1))
        # out = torch.sigmoid(self.conv3(x1))
        out = self.conv3(x1)
        return out


class FCT3D_MODIFY(nn.Module):
    def __init__(self):
        super().__init__()

        # attention heads and filters per block
        att_heads = [2, 2, 2, 2, 2, 2, 2, 2, 2]
        filters = [8, 16, 32, 64, 128, 64, 32, 16, 8]

        # number of blocks used in the model
        blocks = len(filters)

        stochastic_depth_rate = 0.0

        # probability for each block
        dpr = [x for x in np.linspace(0, stochastic_depth_rate, blocks)]

        self.drp_out = 0.3

        # Multi-scale input
        self.scale_img = nn.AvgPool3d(2, 2)

        # model
        self.block_1 = Block_encoder_bottleneck("first", 1, filters[0], att_heads[0], dpr[0])
        self.block_2 = Block_encoder_bottleneck("second", filters[0], filters[1], att_heads[1], dpr[1])
        self.block_3 = Block_encoder_bottleneck("third", filters[1], filters[2], att_heads[2], dpr[2])
        self.block_4 = Block_encoder_bottleneck("fourth", filters[2], filters[3], att_heads[3], dpr[3])
        self.block_5 = Block_encoder_bottleneck("bottleneck", filters[3], filters[4], att_heads[4], dpr[4])
        self.block_6 = Block_decoder(filters[4], filters[5], att_heads[5], dpr[5])
        self.block_7 = Block_decoder(filters[5], filters[6], att_heads[6], dpr[6])
        self.block_8 = Block_decoder(filters[6], filters[7], att_heads[7], dpr[7])
        self.block_9 = Block_decoder(filters[7], filters[8], att_heads[8], dpr[8])


        # self.ds7 = DS_out(filters[6], 2)
        # self.ds8 = DS_out(filters[7], 2)
        # self.ds9 = DS_out(filters[8], 2)
        self.upsample = nn.Upsample(scale_factor=2)
        self.out = nn.Conv3d(filters[8],2, kernel_size=1, stride=1)

    def forward(self, x):
        # Multi-scale input
        scale_img_2 = self.scale_img(x) # (1,1,64,64,64)
        scale_img_3 = self.scale_img(scale_img_2) #(1,1,32,32,32)
        scale_img_4 = self.scale_img(scale_img_3) #(1,1,16,16,16)

        x = self.block_1(x) #(1,8,64,64,64)
        # print(f"Block 1 out -> {list(x.size())}")
        skip1 = x
        # skip1 = self.skip_mamba_1(x) #(1,8,64,64,64)
        x = self.block_2(x, scale_img_2) #(1,16,32,32,32)
        # print(f"Block 2 out -> {list(x.size())}")
        skip2 = x
        # skip2 = self.skip_mamba_2(x) #(1,16,32,32,32)
        x = self.block_3(x, scale_img_3)
        # print(f"Block 3 out -> {list(x.size())}")
        skip3 = x
        # skip3 = self.skip_mamba_3(x) #(1,32,16,16,16)
        x = self.block_4(x, scale_img_4)
        # print(f"Block 4 out -> {list(x.size())}")
        skip4 = x
        # skip4 = self.skip_mamba_4(x) #(1,64,8,8,8)
        x = self.block_5(x)
        # x = self.skip_mamba_5(self.block_5(x)) #(1,128,4,4,4)
        # print(f"Block 5 out -> {list(x.size())}")
        x = self.block_6(x, skip4) #(1,64,8,8,8)
        # print(f"Block 6 out -> {list(x.size())}")
        x = self.block_7(x, skip3) #(1,32,16,16,16)
        # print(f"Block 7 out -> {list(x.size())}")
        skip7 = x #(1,32,16,16,16)
        x = self.block_8(x, skip2) #(1,16,32,32,32)
        # print(f"Block 8 out -> {list(x.size())}")
        skip8 = x #(1,16,32,32,32)
        x = self.block_9(x, skip1) #(1,8,64,64,64)
        # print(f"Block 9 out -> {list(x.size())}")
        skip9 = x #(1,8,64,64,64)

        # out7 = self.ds7(skip7)
        # # print(f"DS 7 out -> {list(out7.size())}")
        # out8 = self.ds8(skip8)
        # print(f"DS 8 out -> {list(out8.size())}")
        # out9 = self.ds9(skip9) # (1,2,128,128,128)
        # print(f"DS 9 out -> {list(out9.size())}")
        out9 = self.out(self.upsample(skip9))

        # return out7, out8, out9

        return out9

def init_weights(m):
    """
    Initialize the weights
    """
    if isinstance(m, nn.Conv3d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


if __name__ == '__main__':
    # https://blog.csdn.net/qq_45041871/article/details/129295325
    import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    import torch
    from torchsummary import summary
    from thop import profile
    from thop import clever_format
    from torch import autocast

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FCT3D_MODIFY().to(device)
    model.apply(init_weights)
    criterion = nn.MSELoss()
    input_shape = (1, 1, 128, 128, 128)  # (batch_size, channels, height, width, depth)
    input = torch.randn(input_shape).to(device)
    print("input_shape: ", input.shape)
    output = model(input)
    with autocast(device.type, enabled=True):
        output = model(input)
        loss = criterion(output, torch.ones_like(output).to(device))  # 假设我们的目标是预测一个全1的输出
        loss.backward()
    print("output_shape: ", output.shape)
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")  # 格式化显示FLOPs和参数量
    print(f"FLOPs: {flops}, Params: {params}")
    # summary(model, input_size=(1, 512, 512), device=device.type)
    # print(model)
    # print(model.state_dict().keys())
    # print(model.state_dict().keys())

    import torch.autograd.profiler as profiler

    with profiler.profile(use_cuda=True) as prof:
        output = model(input)
        target=torch.ones_like(output).to(device)
        loss = criterion(output, target)
        loss.backward()
    # 打印所有事件的统计信息
    print(prof)