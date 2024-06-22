import math
import torch
import torch.nn as nn
from .image_encoder import ImageEncoderViT
from .mask_decoder import Mask_decoder
from .auto_prompter_3D import SPG
from einops import rearrange
from functools import partial

class Gate(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.dim = in_dim
        self.linear = nn.Linear(in_dim, 1)
        self.reset_parameters()

    def forward(self, x):
        x = self.linear(x)
        x = torch.sigmoid(x)
        x = rearrange(x, 'b h w c -> b (h w) c')
        return torch.mean(x, dim=1).unsqueeze(1).unsqueeze(2)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

class _Parallel_Adapter(nn.Module):
    def __init__(self, mlp, adapter_dim):
        super().__init__()
        assert adapter_dim > 0
        self.mlp = mlp
        self.in_dim = self.mlp.lin1.in_features
        self.adapter_dim = adapter_dim
        self.proj_down = nn.Linear(self.in_dim, self.adapter_dim)
        self.nolinear = nn.ReLU()
        self.proj_up = nn.Linear(self.adapter_dim, self.in_dim)
        self.gate = Gate(self.in_dim)
        self.reset_parameters()

    def forward(self, x):
        residual = x
        x_mlp = self.mlp(x)
        x = self.proj_down(x)
        x = self.nolinear(x)
        x = self.proj_up(x)  #(b, n, c)
        x = self.gate(residual) * x
        return x + x_mlp

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.proj_down.weight)
        nn.init.xavier_uniform_(self.proj_up.weight)
        nn.init.constant_(self.proj_down.bias, 0)
        nn.init.constant_(self.proj_up.bias, 0)


class _LoRA_qkv(nn.Module):
    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
            convlora=False
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.gate = Gate(self.dim)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += self.gate(x)*new_q
        qkv[:, :, :, -self.dim:] += self.gate(x)*new_v #加入门控机制
        return qkv


class AIS_SAM(nn.Module):
    def __init__(self, r: int=4, adapter_dim: int=64, spg=SPG(), d_size=20, **kwargs):
        super(AIS_SAM, self).__init__()
        image_encoder = ImageEncoderViT(
            depth=12,
            embed_dim=768,
            img_size=256,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=12,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[2, 5, 8, 11],
            window_size=14,
            out_chans=256,
            d_size=d_size,
        )


        check_path = "/home/tanxz/Codes/3DSAM-adapter/3DSAM-adapter/ais_sam/sam_med2d/checkpth/sam-med2d_b.pth"
        model_params = torch.load(check_path, map_location="cpu")["model"]
        for k in list(model_params.keys()):
            if k.startswith("image_encoder"):
                model_params[k[len("image_encoder."):]] = model_params[k]
            del model_params[k]
        msg = image_encoder.load_state_dict(model_params, strict=False)
        self.image_encoder = image_encoder
        # state_dict = torch.load(checkpoint, map_location='cpu')
        # msg = image_encoder.load_state_dict(state_dict['model'], strict=False)
        print(msg)
        self.w_As = []
        self.w_Bs = []
        # Frozen foundation model ViT encoder
        for n, value in image_encoder.named_parameters():
            value.requires_grad = False   #首先冻结所有参数            decoder和组合微调都是在后面加上的，所以参数必定是需要梯度的
            if 'depth_branch' in n:
                value.requires_grad = True
            if 'fuse_mlp' in n:
                value.requires_grad = True
            if 'neck' in n:
                value.requires_grad = True
            if "adapter" in n:
                value.requires_grad = True
            if "fuse_conv" in n:
                value.requires_grad = True

        for t_layer_i, blk in enumerate(image_encoder.blocks):
            w_qkv_linear = blk.attn.qkv
            mlp = blk.mlp
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )
            blk.mlp = _Parallel_Adapter(mlp, adapter_dim)
        self.reset_parameters()
        self.image_encoder = image_encoder
        self.mask_decoder = Mask_decoder(in_channels=1, out_channels=2, d_size=d_size)
        if spg != None:
            self.spg = spg
        else:
            self.spg = None

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)


    def forward(self, batched_input: torch.Tensor):
        if batched_input.shape[1] == 1:
            batched_input = batched_input.repeat(1, 3, 1, 1, 1)
        if batched_input.shape[1] == 2:
            semantic_prompts = self.spg(batched_input[:, 0, :, :, :].unsqueeze(1), batched_input[:, 1, :, :, :].unsqueeze(1))
            semantic_prompts = rearrange(semantic_prompts, 'b c d h w -> (b d) c h w')
            batched_input = batched_input[:, 0, :, :, :].unsqueeze(1)
            batched_input = batched_input.repeat(1, 3, 1, 1, 1)

        # batched_input = rearrange(batched_input, 'b c d h w -> (b d) c h w')
        # batched_input = batched_input.squeeze(0).permute(1, 0, 2, 3)
        b, c, d, h, w = batched_input.shape
        batched_input = batched_input.permute(0, 2, 1, 3, 4)  #(b, d, c, h, w)
        batched_input = batched_input.contiguous().view(b*d, c, h, w)
        img_embedding, hidden_output = self.image_encoder(batched_input)
        hidden_output[-1] = torch.cat([img_embedding, semantic_prompts], dim=1).permute(0, 2, 3, 1)
        pred = self.mask_decoder(hidden_output[0], hidden_output, b, d)
        return pred


