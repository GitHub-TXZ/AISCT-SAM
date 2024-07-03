from Three_d.uxnet_3d import UXNET_3D
if __name__ == '__main__':
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    import torch
    from torchsummary import summary
    from thop import profile
    from thop import clever_format
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UXNET_3D(
        in_chans=1,
        # out_chans= 3,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        spatial_dims=3,
    ).to(device)
    # input_shape = (1, 1, 128, 128, 128)  # (batch_size, channels, height, width, depth)
    input_shape = (1, 1, 32, 256, 224)
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