import os

def rename_nii_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.nii.gz'):
                file_path = os.path.join(root, file)
                file_name_without_ext = file_path.split('/')[-1][:-7]
                new_file_name = f"{file_name_without_ext}_0000.nii.gz"
                new_file_path = os.path.join(root, new_file_name)
                os.rename(file_path, new_file_path)
                print(f"Renamed {file} to {new_file_name}")

base = "../../../nnUNet_Database/external_val/APIS/imagesTs"
rename_nii_files(base)


