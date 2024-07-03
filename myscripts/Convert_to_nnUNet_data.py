import os
os.environ['nnUNet_raw'] = '/home/tanxz/nnUNet_Database/nnUNet_raw'
os.environ['nnUNet_preprocessed'] = '/home/tanxz/nnUNet_Database/nnUNet_preprocessed'
os.environ['nnUNet_results'] = '/home/tanxz/nnUNet_Database/nnUNet_results'
import multiprocessing
import shutil
from multiprocessing import Pool

import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from utils import find_files_with_extension

def copy_BraTS_segmentation_and_convert_labels_to_nnUNet(in_file: str, out_file: str) -> None:
    # use this for segmentation only!!!
    # nnUNet wants the labels to be continuous. BraTS is 0, 1, 2, 4 -> we make that into 0, 1, 2, 3
    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)

    uniques = np.unique(img_npy)
    for u in uniques:
        if u not in [0, 1, 2]:
            raise RuntimeError('unexpected label')

    seg_new = np.zeros_like(img_npy)
    seg_new[img_npy > 0] = 1
    img_corr = sitk.GetImageFromArray(seg_new)
    img_corr.CopyInformation(img)
    sitk.WriteImage(img_corr, out_file)

if __name__ == '__main__':
    brats_data_dir = '/home/tanxz/Datasets/CPAISD'

    task_id = 22
    task_name = "CPAISD"

    foldername = "Dataset%03.0d_%s" % (task_id, task_name)

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    case_ids = find_files_with_extension(brats_data_dir, "images.nii.gz")

    for c in case_ids:
        file_name  = c.split('/')[-1][:-14]
        shutil.copy(c, join(imagestr, file_name + '_0000.nii.gz'))

    case_ids = find_files_with_extension(brats_data_dir,"masks.nii.gz")
    for c in case_ids:
        file_name  = c.split('/')[-1][:-13]
        copy_BraTS_segmentation_and_convert_labels_to_nnUNet(c, join(labelstr,  file_name + '.nii.gz'))

    generate_dataset_json(out_base,
                          channel_names={0: 'CT'},
                          labels={
                              'background': 0,
                              'infarct lesion': 1,
                          },
                          num_training_cases=len(case_ids),
                          file_ending='.nii.gz',
                          license='txz',
                          reference='txz',
                          dataset_release='1.0')
