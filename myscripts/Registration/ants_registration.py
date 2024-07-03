import os
import glob
from multiprocessing import Pool
from functools import partial
from utlis.registration import registration_to_MNI, augment_symmetric_modality


def get_files(folder):
    all_files = []
    for subdir in os.listdir(folder):
        file_dir = os.path.join(folder, subdir)
        if not os.path.isdir(file_dir):
            continue
        pat_files = glob.glob(f'{file_dir}/mCTA*left.nii.gz')
        for i in pat_files:
            assert os.path.isfile(i), f"{i} does not exist!"
        all_files.append(sorted(pat_files))
    return all_files


if __name__ == '__main__':
    data_path = '/home/wyh/Dataset/ProVe-Test'
    out_dir = "/home/wyh/Dataset/ProVe/"
    file_lists = get_files(data_path)
    pool = Pool(8)
    pool.map(augment_symmetric_modality, file_lists)
    print('All done!!!')
