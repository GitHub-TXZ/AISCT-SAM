import json
import random
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
# 指定数据集的目录
dataset_dir = '/home/tanxz/nnUNet_Database/nnUNet_preprocessed/Dataset019_ISLES22_ATLAS_2/gt_segmentations'
# 定义计算体积的函数
def calculate_volume(data,img):
    return np.sum(data == 1) * (img.header.get_zooms()[0] * img.header.get_zooms()[1] * img.header.get_zooms()[2]) / 1000

# 指定JSON文件的路径
json_file_path = '/home/tanxz/nnUNet_Database/nnUNet_preprocessed/Dataset019_ISLES22_ATLAS_2/splits_final.json'
random.seed(42)
# 打开并读取JSON文件
with open(json_file_path, 'r') as file:
    data = json.load(file)
    fold0 = data[0]
    train = fold0['train']
    random.shuffle(train)
    train_subset1 = train[:459]
    train_subset2 = train[459:524]
    val = fold0['val']

volumes_train_subset1 = []
volumes_train_subset2 = [  ]
volumes_val = []

for filename in train_subset1:
    full_path = f'{dataset_dir}/{filename}.nii.gz'
    img = nib.load(full_path)
    data = img.get_fdata()
    volume = calculate_volume(data,img)
    volumes_train_subset1.append(volume)

for filename in train_subset2:
    full_path = f'{dataset_dir}/{filename}.nii.gz'
    img = nib.load(full_path)
    data = img.get_fdata()
    volume = calculate_volume(data,img)
    volumes_train_subset2.append(volume)

for filename in val:
    full_path = f'{dataset_dir}/{filename}.nii.gz'
    img = nib.load(full_path)
    data = img.get_fdata()
    volume = calculate_volume(data, img)
    volumes_val.append(volume)

plt.rcParams.update({'font.size': 14})
# 绘制散点图
plt.scatter(range(len(volumes_train_subset1)), volumes_train_subset1, label='Training', color='blue')
plt.xlabel('Sample Index')
plt.ylabel('Ground Truth Volume (mL)')
plt.legend()
plt.show()
plt.close()

plt.scatter(range(len(volumes_train_subset2)), volumes_train_subset2, label='Validation', color='red')
plt.xlabel('Sample Index')
plt.ylabel('Ground Truth Volume (mL)')
plt.legend()
plt.show()
plt.close()

plt.scatter(range(len(volumes_val)), volumes_val, label='Testing', color='green')
plt.xlabel('Sample Index')
plt.ylabel('Ground Truth Volume (mL)')
plt.legend()
plt.show()
plt.close()

