import math
import torch
import torch.nn as nn
from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from einops import rearrange
from functools import partial
from .transformer import TwoWayTransformer
from torch.nn import functional as F
from .auto_prompter_3D import SPG
from typing import Type
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



def state_dict_modify(state_dict, image_size, vit_patch_size):
    except_keys = ['mask_tokens', 'output_hypernetworks_mlps', 'iou_prediction_head'] #去除这些key，模型变成语义分割的模型
    state_dict = {k: v for k, v in state_dict.items() if
                      k in state_dict.keys() and except_keys[0] not in k and except_keys[1] not in k and except_keys[2] not in k}
    new_state_dict = {k: v for k, v in state_dict.items() if
                      k in state_dict.keys() and except_keys[0] not in k and except_keys[1] not in k and except_keys[2] not in k}
    pos_embed = new_state_dict['image_encoder.pos_embed']
    token_size = int(image_size // vit_patch_size)
    if pos_embed.shape[1] != token_size:
        # resize pos embedding, which may sacrifice the performance, but I have no better idea
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
        new_state_dict['image_encoder.pos_embed'] = pos_embed
        rel_pos_keys = [k for k in state_dict.keys() if 'rel_pos' in k]
        global_rel_pos_keys = [k for k in rel_pos_keys if '2' in k or '5' in  k or '8' in k or '11' in k]
        for k in global_rel_pos_keys:
            rel_pos_params = new_state_dict[k]
            h, w = rel_pos_params.shape
            rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
            rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
            new_state_dict[k] = rel_pos_params[0, 0, ...]
    state_dict.update(new_state_dict) # 更新state_dict
    return state_dict


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(input_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, output_dim)
        self.act = act()
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.constant_(self.lin1.bias, 0)
        nn.init.xavier_uniform_(self.lin2.weight)
        nn.init.constant_(self.lin2.bias, 0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm2(self.lin2(self.act(self.lin1(self.norm1(x)))))


class ICASSP(nn.Module):
    def __init__(self, r: int=4, adapter_dim: int=64, d_size=20, **kwargs):
       
        super(ICASSP, self).__init__()
        self.spg = SPG()
        num_classes = 1   # 去掉背景还剩几类
        self.num_classes = num_classes
        prompt_embed_dim = 256
        vit_patch_size = 16
        image_size = 256
        self.image_size = image_size
        image_embedding_size = image_size // vit_patch_size
        image_encoder = ImageEncoderViT(
            depth=12,
            embed_dim=768,
            img_size=image_size,
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
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        )
        mask_decoder=MaskDecoder(
            num_multimask_outputs=num_classes,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=num_classes,
            iou_head_hidden_dim=256,
        )
        self.token_mlp = MLP(
            input_dim=768,
            output_dim=256,
            mlp_dim=384,
            act=nn.GELU,
        )
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder

        check_path = "/repository/users/tanxz/model_checkpoint/sam_vit_b_01ec64.pth"
        model_params = torch.load(check_path, map_location="cpu")
        model_params = state_dict_modify(model_params, image_size, vit_patch_size)
        
        
        for k in list(model_params.keys()):
            if k.startswith("image_encoder"):
                model_params[k[len("image_encoder."):]] = model_params[k]
            del model_params[k]
        msg = image_encoder.load_state_dict(model_params, strict=False)
        print(msg)
        self.image_encoder = image_encoder
        
        
        check_path = "/repository/users/tanxz/model_checkpoint/sam_vit_b_01ec64.pth"
        model_params = torch.load(check_path, map_location="cpu")
        for k in list(model_params.keys()):
            if k.startswith("prompt_encoder"):
                model_params[k[len("prompt_encoder."):]] = model_params[k]
            del model_params[k]
        msg = prompt_encoder.load_state_dict(model_params, strict=False)
        print(msg)
        self.prompt_encoder = prompt_encoder
        
        check_path = "/repository/users/tanxz/model_checkpoint/sam_vit_b_01ec64.pth"
        model_params = torch.load(check_path, map_location="cpu")
        model_params = state_dict_modify(model_params, image_size, vit_patch_size)
        for k in list(model_params.keys()):
            if k.startswith("mask_decoder"):
                model_params[k[len("mask_decoder."):]] = model_params[k]
            del model_params[k]
        msg = mask_decoder.load_state_dict(model_params, strict=False)
        print(msg)
        self.mask_decoder = mask_decoder
        
        del model_params
        # state_dict = torch.load(checkpoint, map_location='cpu')
        # msg = image_encoder.load_state_dict(state_dict['model'], strict=False)
       
        self.w_As = []
        self.w_Bs = []
        
         # Frozen foundation model ViT encoder
        for n, value in image_encoder.named_parameters():
            value.requires_grad = False   #首先冻结所有参数   decoder和组合微调都是在后面加上的，所以参数都是可训练的
            if 'depth_branch' in n:
                value.requires_grad = True
            if 'neck' in n:
                value.requires_grad = True
            if "patch_embed" in n:
                value.requires_grad = True
            # if "adapter" in n:
            #     value.requires_grad = True
            if 'token_mlp' in n:
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

        
    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)


    def forward(self, batched_input):  # 通过外部传入的参数决定是否输出多个mask
        # input_images = self.preprocess(batched_input)

        b, c, d, h, w = batched_input.shape # c=2
        sp = self.spg(batched_input[:, 0, :, :, :].unsqueeze(1), batched_input[:, 1, :, :, :].unsqueeze(1))
        sp = sp.permute(0, 2, 1, 3, 4) .contiguous().view(b*d, 1, h//4, w//4)
        
        image_embeddings, diff_tokens = self.image_encoder(batched_input)
        diff_tokens = diff_tokens + self.image_encoder.pos_embed # 加上位置编码
        diff_tokens = rearrange(diff_tokens, 'b h w c -> b (h w) c')
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None, boxes=None, masks=sp   # torch.ones(b*d, 1, h//4, w//4).cuda()
        )
        # 替换 sparse_embeddings sparse_embeddings 
        sparse_embeddings =  self.token_mlp(diff_tokens)
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings= image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True #这个参数已经没有用了
        )
        masks = self.postprocess_masks(
            low_res_masks,
            input_size=(self.image_size, self.image_size),
            original_size=(self.image_size, self.image_size)
        )
        outputs = {
            'masks': masks,
            'iou_predictions': iou_predictions,
            'low_res_logits': low_res_masks
        }
        return outputs['masks'].view(b, d, self.num_classes+1, h, w).permute(0, 2, 1, 3, 4).contiguous(), sp, iou_predictions  #train
        # return outputs['masks'].view(b, d, self.num_classes+1, h, w).permute(0, 2, 1, 3, 4).contiguous() #test

    def postprocess_masks(
            self,
            masks: torch.Tensor,
            input_size: Tuple[int, ...],
            original_size: Tuple[int, ...],
        ) -> torch.Tensor:
            """
            Remove padding and upscale masks to the original image size.

            Arguments:
            masks (torch.Tensor): Batched masks from the mask_decoder,
                in BxCxHxW format.
            input_size (tuple(int, int)): The size of the image input to the
                model, in (H, W) format. Used to remove padding.
            original_size (tuple(int, int)): The original size of the image
                before resizing for input to the model, in (H, W) format.

            Returns:
            (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
                is given by original_size.
            """
            masks = F.interpolate(
                masks,
                (self.image_encoder.img_size, self.image_encoder.img_size),
                mode="bilinear",
                align_corners=False,
            )
            masks = masks[..., : input_size[0], : input_size[1]]
            masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
            return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
if __name__ == "__main__":
    import torch
    import os
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from torchvision.transforms import functional as TF
    vis_dir = 'vis'
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    upsampled_sp = torch.nn.functional.interpolate(sp.cpu(), size=(20, 256, 256), mode='trilinear', align_corners=False)
    for i in range(upsampled_sp.shape[2]):
        img = upsampled_sp[0, 0, i, :, :]
        img = img - img.min()
        img = img / img.max()
        img_np = img.numpy()
        img_jet = cm.jet(img_np)  # 这将返回一个RGB图像
        plt.imsave(os.path.join(vis_dir, f'{i+1}.png'), img_jet, cmap='jet')
    print("Images saved successfully.")