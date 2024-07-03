import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from Two_d.isnet import ISNet
if __name__ == '__main__':
    #Yang H, Huang C, Nie X, et al.
    # Is-Net: automatic ischemic stroke lesion
    # segmentation on CT images[J].
    # IEEE Transactions on Radiation and Plasma Medical
    # Sciences, 2023.
    import torch
    from torchsummary import summary
    from thop import profile
    from thop import clever_format
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ISNet(3, 2).to(device)
    input_shape = (2, 3, 512, 512)  # (batch_size, channels, height, width, depth)
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
