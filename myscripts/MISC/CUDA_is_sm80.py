# import torch
#
# print("python.__version =", torch.__version__)
# print("device =", torch.cuda.get_device_name())
#
# # d_models=[384, 512, 560, 576, 640, 1024, 1400]
# d_models=[8, 16, 32, 64, 128, 256, 512,1024,2048,4096]
#
# for d_model in d_models:
#     try:
#         print('d_model =', d_model, end='')
#         t = torch.nn.Transformer(d_model=d_model, batch_first=True).cuda()
#         x = torch.randn(1, 10, d_model).cuda()
#         with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
#             t(x, x).mean().backward()
#         print(" -> OK")
#     except RuntimeError as e:
#         print(" -> " + str(e))



# import torch
#
# t = torch.nn.Transformer(d_model=576, batch_first=True).cuda() #512时正常
# x = torch.randn(1, 10, 576).cuda() #512时正常
# with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
#     t(x, x).mean().backward()


#
# import torch
# import torch.nn as nn
# class MyModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         self.q = nn.Parameter(torch.rand(1, 64, 64, 96, dtype=torch.float16).to("cuda"))
#         self.k = nn.Parameter(torch.rand(1, 64, 64, 96, dtype=torch.float16).to("cuda"))
#         self.v = nn.Parameter(torch.rand(1, 64, 64, 96, dtype=torch.float16).to("cuda"))
#
#     def forward(self, x):
#         return torch.nn.functional.scaled_dot_product_attention(self.q, self.k, self.v, is_causal=True,
#                                                                 dropout_p=0.0) * x
# model = MyModel()
# with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
#     inp = torch.Tensor([3.]).to("cuda", torch.float16)
#     res = model(inp)
#     loss = torch.sum(res)
#     loss.backward()


# python -c "import torch;print(torch.__config__.show(), torch.cuda.get_device_properties(0))"










#
#
# import torch
# import torch.autograd.profiler as profiler
#
# with profiler.profile(use_cuda=True) as prof:
#     output = model(input)
#     loss = criterion(output, target)
#     loss.backward()
#
# # 打印所有事件的统计信息
# print(prof)
#
#
#
#
# output = model(input)
# loss = criterion(output, target)
# # 在反向传播之前打印权重和梯度
# for name, param in model.named_parameters():
#     if 'specific_layer_name' in name:  # 替换为特定层的名称
#         print(name, param.data)  # 打印权重
#         print(name, param.grad)  # 打印梯度，如果None，说明没有梯度
#
#
#
#
#
# def print_gradient_hook(grad):
#     print(grad)
#
# # 在特定层的权重上注册一个反向传播钩子
# model.specific_layer_name.weight.register_hook(print_gradient_hook)
#
# output = model(input)
# loss = criterion(output, target)
# loss.backward()
#
#
#
#
#
# def print_intermediate_output_hook(module, input, output):
#     print(output)
#
# # 给特定层添加一个前向传播钩子
# hook = model.specific_layer_name.register_forward_hook(print_intermediate_output_hook)
#
# output = model(input)
# loss = criterion(output, target)
# loss.backward()
#
# # 如果不再需要钩子，可以移除它
# hook.remove()
#
#
#
# from torchsummary import summary
#
# summary(model, input_size=(1, 3, 224, 224))
#
#
#
#


import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True
data = torch.randn([25, 128, 63, 63], dtype=torch.float, device='cuda', requires_grad=True)
net = torch.nn.Conv2d(128, 128, kernel_size=[2, 2], padding=[1, 1], stride=[1, 1], dilation=[1, 1], groups=1)
net = net.cuda().float()
out = net(data)
out.backward(torch.randn_like(out))
torch.cuda.synchronize()













