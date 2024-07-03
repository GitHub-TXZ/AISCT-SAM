from typing import Tuple, Union

import torch
import torch.nn as nn

from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock


class Mask_decoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            feature_size: int = 16,
            hidden_size: int = 768,
            norm_name: Union[Tuple, str] = "instance",
            conv_block: bool = False,
            res_block: bool = True,
            d_size: int = 25,

    ) -> None:
        super().__init__()

        self.feat_size = (
            d_size,
            16,
            16,
        )
        self.hidden_size = hidden_size

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=(1, 2, 2),
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=(1, 2, 2),
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=(1, 2, 2),
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size // 3,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=(1, 2, 2),
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=(1, 2, 2),
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=(1, 2, 2),
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=(1, 2, 2),
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)  # type: ignore

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.contiguous().view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x


    def forward(self, hidden_states_out, semantic_prompts, b, d):
        hidden_states_out = [i.contiguous().view(b, d, i.shape[1],i.shape[2],i.shape[3]).permute(0, 4, 1, 2, 3) for i in hidden_states_out]

        # enc1 由Semantic Prompts 产生
        x1= semantic_prompts[0]
        enc1 = self.encoder1(x1)

        x2 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(x2, self.hidden_size, self.feat_size))
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3, self.hidden_size, self.feat_size))
        x4 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(x4, self.hidden_size, self.feat_size))

        # Add Semantic Prompts
        enc2 = enc2 + semantic_prompts[1]
        enc3 = enc3 + semantic_prompts[2]
        enc4 = enc4 + semantic_prompts[3]

        dec4 = hidden_states_out[-1] + semantic_prompts[4]

        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        logits = self.out(out)
        return logits


if __name__ == '__main__':
    import os
    b = 1
    d = 40
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    net = Mask_decoder(in_channels=1, out_channels=2, d_size=40)
    print(net)
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total parameters: {total_params / 1e6}M")
    x = torch.randn(1, 1, 40, 256, 256)
    fake = torch.randn(b*d, 16, 16 ,768)
    hidden_states_out = [torch.randn(b*d, 256, 256, 3)] + [fake for i in range(11)] + [torch.randn(b*d,16,16,256)]
    semantic_prompts = [torch.randn(b,1,d,256,256), torch.randn(b,32,d,128,128),torch.randn(b,64,d,64,64),torch.randn(b,128,d,32,32),torch.randn(b,256,d,16,16)]
    y = net(hidden_states_out, semantic_prompts, b=1, d=40)
    print(y.shape)
