import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from monai.networks.nets import SwinUNETR


# 子类父类同名是可以的
class Swin_UNETR(SwinUNETR):
    def __init__(self, img_size=(128, 128, 128),in_channels=1,out_channels=2,feature_size=48,use_checkpoint=True):
        super(Swin_UNETR, self).__init__(img_size=img_size,
                      in_channels=in_channels,
                      out_channels=out_channels,
                      feature_size=feature_size,
                      use_checkpoint=use_checkpoint,
                      )
    def forward(self, x_in):
        out = super().forward(x_in)
        return out


if __name__ == '__main__':
    # F.pad() 函数的填充是从后往前逐渐填充的， x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1)) (左右、上下、前后)
    import torch
    from torchsummary import summary
    from thop import profile
    from thop import clever_format
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Swin_UNETR().to(device)
    # input_shape = (1, 1, 64, 320, 256)  # (batch_size, channels, height, width, depth)
    input_shape = (1, 1, 128, 128, 128)
    input= torch.randn(input_shape).to(device)
    print("input_shape: ", input.shape)
    output = model(input)
    print("output_shape: ", output.shape)
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")  # 格式化显示FLOPs和参数量
    print(f"FLOPs: {flops}, Params: {params}")
    # summary(model, input_size=(1, 512, 512), device=device.type)
    # print(model)
    # print(model.state_dict().keys())
    # print(model.state_dict().keys())
