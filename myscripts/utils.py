import os
import nibabel as nib
import numpy as np


def find_files_with_extension(directory, extension):
    """
    递归查找目录中指定后缀名的所有文件，并返回它们的绝对路径列表。

    参数:
    - directory: 要搜索的目录路径。
    - extension: 文件的后缀名，例如'.txt', '.jpg'等。

    返回:
    - 符合条件的文件的绝对路径列表。
    """
    result_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                file_path = os.path.join(root, file)
                result_files.append(file_path)

    return result_files

def rename_files_with_prefix(paths):
    """
    给定文件路径列表，将所有文件重命名，在文件扩展名之前加上"_0000"。

    参数:
    - paths: 文件路径列表。

    返回:
    - 无。文件将被重命名。
    """
    for old_path in paths:
        directory, filename = os.path.split(old_path)
        # base_name, extension = os.path.splitext(filename)
        base_name, extension = filename[:-7], filename[-7:]
        new_filename = f"{base_name}_0000{extension}"
        new_path = os.path.join(directory, new_filename)
        os.rename(old_path, new_path)



def process_nii_files(file_paths):
    """
        读取给定的.nii.gz文件列表，并对其进行处理后保存回原文件。

        参数：
        file_paths (list of str): 包含.nii.gz文件路径的列表。

        返回值：
        无。

        示例用法：
        # >>> file_paths = ["/path/to/your/file1.nii.gz", "/path/to/your/file2.nii.gz"]
        # >>> process_nii_files(file_paths)
        """
    for file_path in sorted(file_paths):
        try:
            nii_img = nib.load(file_path)
            data = nii_img.get_fdata()
            header = nii_img.header
            data = np.flip(data, axis=0)
            # data[data > 0] = 1

            modified_nii_img = nib.Nifti1Image(data, affine=nii_img.affine, header=header)
            nib.save(modified_nii_img, file_path.replace("0000","0001"))

            print(f"Processed {file_path} successfully.")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    # 示例用法
    directory_path = '/home/tanxz/nnUNet_Database/nnUNet_raw/Dataset023_ATLAS_Sym_basedon_D19/imagesTr'
    extension = '0000.nii.gz'
    found_files = find_files_with_extension(directory_path, extension)
    # rename_files_with_prefix(found_files)
    process_nii_files(found_files)
