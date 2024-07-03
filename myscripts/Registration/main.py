import os
from pathlib import Path
from multiprocessing import Pool
from utlis.registration import registration_to_MNI, registration_to_NCCT


def main(file_list, is_label=False):
    pass


if __name__ == '__main__':
    data_dir = Path('/homeb/wyh/Datasets/Stroke')
    CTA_list = [i.as_posix() for i in data_dir.rglob('CTA_brain.nii.gz')]
    p = Pool(8)
    p.map(registration_to_NCCT, CTA_list)
    p.close()
    p.join()
