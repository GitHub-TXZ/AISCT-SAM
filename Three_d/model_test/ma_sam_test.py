from Three_d.ma_sam.sam_fact_tt_image_encoder import Fact_tt_Sam
from Three_d.ma_sam.segment_anything import sam_model_registry
from Three_d.ma_sam import MA_SAM
import os
import torch
# register model
checkpoint = "/home/tanxz/Model_Checkpoint/sam_vit_b_01ec64.pth"
sam, img_embedding_size = sam_model_registry['vit_b'](image_size=512,
                                                            num_classes=1,
                                                            checkpoint=checkpoint, pixel_mean=[0., 0., 0.],
                                                            pixel_std=[1., 1., 1.])


if __name__ == '__main__':
    import torch.autograd.profiler as profiler
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    # net = Fact_tt_Sam(sam, 64, s=1.0).cuda()
    # net = Fact_tt_Sam(sam, 64, s=1.0).cuda()
    net = MA_SAM(sam, 64, s=1.0).cuda()

    # with profiler.profile(profile_memory=True, record_shapes=True, use_cuda=True) as prof:
    #     y = net(x, False, 512)
    # print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

    x = torch.randn(1, 1, 5, 512, 512).to('cuda')
    y = net(x, False, 512)
    total_allocated_memory_mib = torch.cuda.memory_allocated() / (1024 ** 2)
    max_memory_allocated_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"Total Allocated GPU Memory: {total_allocated_memory_mib:.2f} MiB")
    print(f"Max GPU Memory Usage: {max_memory_allocated_mib:.2f} MiB")
    print(y.shape)