if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    import torch
    from torchsummary import summary
    from Two_d.utnet import UTNet
    from thop import profile
    from thop import clever_format
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # UTNet论文中的输入是256*256
    model = UTNet(in_chan=1, base_chan=32).to(device)
    # input_shape = (1, 1, 256*2, 256*2)  # (batch_size, channels, height, width, depth)
    input_shape = (1, 1, 256, 256)
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

