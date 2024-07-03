
from Three_d.highres3dnet import HighRes3DNet
import os
import torch

if __name__ == '__main__':
    import torch
    from torchsummary import summary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HighRes3DNet(in_channels=1, out_channels=3).to(device)
    # 生成伪3D数据测试模型
    input_shape = (1, 1, 64, 64, 64)  # (batch_size, channels, height, width, depth)
    input = torch.randn(input_shape).to(device)
    print("input_shape: ", input.shape)
    output = model(input)
    print("output_shape: ", output.shape)
    summary(model, input_size=(1, 64, 64, 64), device=device.type)
    # print(model)
    # print(model.state_dict().keys())
    # print(model.state_dict().keys())