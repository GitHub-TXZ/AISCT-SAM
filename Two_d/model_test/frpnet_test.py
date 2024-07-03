if __name__ == '__main__':
    # A feature enhanced network for stroke lesion segmentation from brain MRI images
    import os
    from Two_d.frpnet import FRPNet
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    import torch
    from torchsummary import summary
    from thop import profile
    from thop import clever_format
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FRPNet().to(device)
    input_shape = (1, 1, 192, 192)  # (batch_size, channels, height, width, depth)
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

