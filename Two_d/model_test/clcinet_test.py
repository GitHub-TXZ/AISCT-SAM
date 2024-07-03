import torch
from Two_d.clcinet import CLCInet
if __name__ == '__main__':
    # FLOPs: 387.945G, Params: 38.513
    # https://blog.csdn.net/qq_37151108/article/details/121992270
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    import torch
    from torchsummary import summary
    from thop import profile
    from thop import clever_format
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CLCInet().to(device)
    input_shape = (8, 1, 256, 224)  # (batch_size, channels, height, width, depth)
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
