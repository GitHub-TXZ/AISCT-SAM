from monai.losses import DiceCELoss, DiceLoss
import surface_distance
from surface_distance import metrics
from medpy.metric import binary
dice_loss = DiceLoss(include_background=False, softmax=False, to_onehot_y=True, reduction="none")
import numpy as np
np.bool = np.bool_

import os
import SimpleITK as sitk

def metrics_compute(path_gt, path_pred):
    files_gt = sorted(set(file for file in os.listdir(path_gt) if file.endswith('.nii.gz')))
    files_pred = sorted(set(file for file in os.listdir(path_pred) if file.endswith('.nii.gz')))
    common_files = set(files_gt).intersection(set(files_pred))
    dice_list = []
    hd95_list = []
    nsd_list = []
    assd_list = []
    if not common_files:
        print("No common .nii.gz files found. using Mode 1 ")
    else:
        print("common .nii.gz files found. using Mode 2")
        files_pred = files_gt = list(common_files)
    for gt_file_name, pred_file_name in zip(files_gt, files_pred):
        file_path_gt = os.path.join(path_gt, gt_file_name)
        file_path_pred = os.path.join(path_pred, pred_file_name)
        try:
            image_gt = sitk.ReadImage(file_path_gt)
            image_pred = sitk.ReadImage(file_path_pred)
            voxel_spacing = image_gt.GetSpacing()
            data_gt = sitk.GetArrayFromImage(image_gt)
            data_pred = sitk.GetArrayFromImage(image_pred)
            # 在这里可以对数据进行进一步处理
            # ...
            print(f"File: {gt_file_name}")
            # print(f"Data from {path_gt}: {data_gt.shape}")
            # print(f"Data from {path_pred}: {data_pred.shape}")
            print("=" * 30)
        except Exception as e:
            print(f"Error reading {pred_file_name}: {str(e)}")
        # dice_list.append(1 - dice_loss(data_pred, data_gt))
        if data_pred.max() == 0 or data_gt.max() == 0:
            dice_list.append(0)
            nsd_list.append(np.nan)
            hd95_list.append(np.nan)
        else:
            dice_list.append(binary.dc(data_gt, data_pred))
            # ssd = surface_distance.compute_surface_distances((data_gt == 1), (data_pred == 1), spacing_mm=voxel_spacing[::-1])
            # assd_list.append(binary.assd((data_gt == 1), (data_pred == 1), voxelspacing=voxel_spacing[::-1]))
            # nsd_list.append(metrics.compute_surface_dice_at_tolerance(ssd, 5))  # kits
            hd95_list.append(binary.hd95((data_pred == 1), (data_gt == 1), voxelspacing=voxel_spacing[::-1]))
        print('done',gt_file_name)

    print(f"Average dice: {np.nanmean(dice_list)}, std: {np.nanstd(dice_list)}")
    print(f"Average hd95: {np.nanmean(hd95_list)}, std: {np.nanstd(hd95_list)}")
    # print(f"Average assd: {np.nanmean(assd_list)}, std: {np.nanstd(assd_list)}")
    # print(f"Average nsd: {np.nanmean(nsd_list)}", f"std: {np.nanstd(nsd_list)}")
    print(f"Total cases: {len(dice_list)}")

if __name__ == '__main__':
    # path_gt = '../../../nnUNet_Database/nnUNet_raw/Dataset018_AISD_Skull_Strip_Different_Spacing/labelsTr'
    # path_pred = "../../../nnUNet_Database/nnUNet_results/Dataset018_AISD_Skull_Strip_Different_Spacing/nnUNetTrainerUMambaBot__nnUNetPlans__3d_fullres/fold_0/validation"
    #
    # path_gt  = "../../../nnUNet_Database/external_val/APIS/labelsTs"
    # path_pred = "../../../nnUNet_Database/external_val/APIS/labelsPr"
    # path_gt = "../../../nnUNet_Database/external_val/CPAISD_Testset/labelsTs"
    # path_pred = "../../../nnUNet_Database/external_val/CPAISD_Testset/labelsPr"

    path_gt = "../../../nnUNet_Database/external_val/PIC_40test/labelsTs"
    path_pred = "../../../nnUNet_Database/external_val/PIC_40test/labelsPr"
    metrics_compute(path_gt, path_pred)




# import torch
# import torch.nn as nn
#
# # 定义深度可分离卷积模块
# class DepthwiseSeparableConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size):
#         super(DepthwiseSeparableConv, self).__init__()
#         self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, groups=in_channels, padding=kernel_size // 2)
#         self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
#
#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         return x
#
# # 使用深度可分离卷积模块
# in_channels = 3
# out_channels = 64
# kernel_size = 3
#
# # 输入特征图尺寸为 (batch_size, in_channels, height, width)
# input_data = torch.randn(1, in_channels, 32, 32)
#
# # 创建深度可分离卷积模块实例
# depthwise_sep_conv = DepthwiseSeparableConv(in_channels, out_channels, kernel_size)
#
# # 对输入数据进行前向传播
# output = depthwise_sep_conv(input_data)
# print(output.shape)  # 输出特征图的形状
