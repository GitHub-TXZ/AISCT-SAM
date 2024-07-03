import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
from Three_d.coma import CoMa
import torch.nn as nn
from torch import autocast
if __name__ == '__main__':
    # F.pad() 函数的填充是从后往前逐渐填充的， x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1)) (左右、上下、前后)
    import torch
    from torchsummary import summary
    from thop import profile
    from thop import clever_format
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CoMa().to(device)
    # input_shape = (1, 1, 128, 240, 192)  # (batch_size, channels, height, width, depth)
    # input_shape = (1, 1, 32, 256, 224)
    input_shape = (2, 1, 64, 240, 192)
    input= torch.randn(input_shape).to(device)
    print("input_shape: ", input.shape)
    criterion = nn.MSELoss()
    # with autocast(device.type, enabled=True):
    output = model(input)
    loss = criterion(output, torch.ones_like(output).to(device))  # 假设我们的目标是预测一个全1的输出
    loss.backward()
    print("output_shape: ", output.shape)
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")  # 格式化显示FLOPs和参数量
    print(f"FLOPs: {flops}, Params: {params}")
    # summary(model, input_size=(1, 512, 512), device=device.type)
    # print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    # print(model)
    # print(model.state_dict().keys())
    # print(model.state_dict().keys())

