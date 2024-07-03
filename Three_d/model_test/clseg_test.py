import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from Three_d.clseg import ClSeg
import torch.nn as nn
from torch import autocast
if __name__ == '__main__':
    # F.pad() 函数的填充是从后往前逐渐填充的， x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1)) (左右、上下、前后)
    import torch
    from torchsummary import summary
    from thop import profile
    from thop import clever_format
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    self.network = ClSeg(in_channels, 16, num_classes,
                         len(pool_op_kernel_sizes[1:]),
                         2, 2, nn.Conv3d, nn.InstanceNorm3d, {'eps': 1e-5, 'affine': True}, nn.Dropout3d,
                         {'p': 0, 'inplace': True},
                         nn.LeakyReLU, {'negative_slope': 1e-2, 'inplace': True}, True, False, lambda x: x,
                         InitWeights_He(1e-2),
                         pool_op_kernel_sizes[1:], conv_kernel_size, False, True,
                         True)


    model = ClSeg().to(device)
    # input_shape = (1, 1, 128, 240, 192)  # (batch_size, channels, height, width, depth)
    input_shape = (1, 1, 16, 320, 320)
    # input_shape = (1, 3, 256, 224)
    # input_shape = (2, 1, 64, 240, 192)
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

