# import os
# import shutil
#
# # 定义源目录和目标目录
# src_images_dir = '/home/tanxz/nnUNet_Database/nnUNet_raw/Dataset013_PIC_remove_invalid/imagesTr'
# src_labels_dir = '/home/tanxz/nnUNet_Database/nnUNet_raw/Dataset013_PIC_remove_invalid/labelsTr'
# dst_dir = '/home/tanxz/nnUNet_Database/external_val/PIC_40test'
#
# # 创建目标子目录
# images_ts_dir = os.path.join(dst_dir, 'imagesTs')
# labels_ts_dir = os.path.join(dst_dir, 'labelsTs')
#
# os.makedirs(images_ts_dir, exist_ok=True)
# os.makedirs(labels_ts_dir, exist_ok=True)
#
# # 定义要复制的文件名模式列表
# filename_patterns = [
#     "SB_09353_1", "SB_13247_1", "SB_14977_1", "SB_09219_1", "SB_11538_1",
#     "SB_10883_1", "SB_10845_1", "SB_04499_1", "SB_07743_1", "SB_04254_1",
#     "SB_10591_1", "SB_11652_1", "SB_05945_1", "SB_06822_1", "SB_14538_1",
#     "SB_09056_1", "SB_05789_1", "SB_09310_1", "SB_09464_1", "SB_08039_1",
#     "SB_11895_1", "SB_12945_1", "SB_05033_1", "SB_08515_1", "SB_12432_1",
#     "SB_05739_1", "SB_06829_1", "SB_08042_1", "SB_12650_1", "SB_03516_2",
#     "SB_10297_1", "SB_12387_1", "SB_10143_1", "SB_05235_1", "SB_06403_1",
#     "SB_07199_1", "SB_03486_1", "SB_13085_1", "SB_12811_1", "SB_05136_1"
# ]
#
# for pattern in filename_patterns:
#     file_name = f"{pattern}"
#     for file in os.listdir(src_images_dir):
#         if file_name in file:
#             src_file_path = os.path.join(src_images_dir, file)
#             dst_file_path = os.path.join(images_ts_dir, file)
#             shutil.copy(src_file_path, dst_file_path)
#
#     for file in os.listdir(src_labels_dir):
#         if file_name in file:
#             src_file_path = os.path.join(src_labels_dir, file)
#             dst_file_path = os.path.join(labels_ts_dir, file)
#             shutil.copy(src_file_path, dst_file_path)
#
# print("Files copied successfully.")


# import ants
# import numpy as np
# import numpy as np
# import matplotlib.pyplot as plt
#
# def display_image_slices(image):
#     if image.ndim != 3:
#         raise ValueError("Input image must be a 3D array")
#     x, y, z = image.shape
#     mid_slice = z // 2
#     fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#     axs[0].imshow(image[:,:,mid_slice], cmap='gray')
#     axs[0].set_title('Original Image Slice')
#     axs[0].axis('off')
#     flipped_image = np.flip(image, axis=1)
#     axs[1].imshow(flipped_image[:,:, mid_slice], cmap='gray')
#     axs[1].set_title('Flipped Image Slice')
#     axs[1].axis('off')
#     plt.show()
#
#
# original_image = ants.image_read('0019983_0000.nii.gz')
# # original_image = original_image.numpy()
# # display_image_slices(original_image)
#
# flipped_image = ants.from_numpy(np.flip(original_image.numpy(), axis=1))
#
#
# registration = ants.registration(fixed=original_image, moving=flipped_image, type_of_transform='rigid')
# warped_image = ants.apply_transforms(fixed=original_image, moving=flipped_image, transformlist=registration['fwdtransforms'])
# ants.plot((original_image, warped_image), rows=[1,2])
# ants.image_write(warped_image, 'warped_image_path.nii.gz')


import cc3d
import numpy as np

# 创建一个三维数组，例如一个全为1的512x512x512数组
labels_in = np.ones((512, 512, 512), dtype=np.int32)

# 进行连通域标记，这里使用26连接性
labels_out = cc3d.connected_components(labels_in, connectivity=26)
gt_data_ori = cc3d.dust(
        labels_out, threshold=30, connectivity=26, in_place=True
    )