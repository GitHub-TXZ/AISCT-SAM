# import os
# import numpy as np
# import nibabel as nib
#
# import os
# import numpy as np
# import nibabel as nib
#
#
# def convert_npz_to_nii(source_dir, destination_dir):
#     if not os.path.exists(destination_dir):
#         os.makedirs(destination_dir)
#     npz_files = [file for file in os.listdir(source_dir) if file.endswith('.npz')]
#     for npz_file in npz_files:
#         npz_path = os.path.join(source_dir, npz_file)
#         npz_data = np.load(npz_path)
#         data = npz_data['probabilities']
#         nii_file = npz_file.replace('.npz', '.nii.gz')
#         nii_path = os.path.join(destination_dir, nii_file)
#         nii = nib.Nifti1Image(data, np.eye(4))  # 创建NIfTI对象
#         nib.save(nii, nii_path)
#         print(f"Converted {npz_file} to {nii_file}.")
#
#

#
# if __name__ == '__main__':
#     source_directory = "/home/tanxz/nnUNet_Database/nnUNet_results/Dataset997_ProveIT/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0_AIS_SAM_3D_v2_lr1e-4/validation"
#     destination_directory = "/home/tanxz/nnUNet_Database/nnUNet_results/Dataset997_ProveIT/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0_AIS_SAM_3D_v2_lr1e-4/prob_softmax"
#     convert_npz_to_nii(source_directory, destination_directory)
