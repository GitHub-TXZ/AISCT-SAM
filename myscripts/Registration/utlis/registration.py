import os
import shutil
import subprocess
import nibabel as nib
import ants
import numpy as np


def registration_to_NCCT(CTA_path):
    print(CTA_path)
    NCCT_path = CTA_path.replace('CTA', 'NCCT')
    save_path = CTA_path.replace('CTA', 'regCTA')
    CTA_img = ants.image_read(CTA_path)
    NCCT_img = ants.image_read(NCCT_path)
    mytx = ants.registration(fixed=NCCT_img, moving=CTA_img, type_of_transform='SyN')
    ants.image_write(mytx['warpedmovout'], save_path)


def registration_to_MNI(file, out_base, prefix='MNI', flag=False):
    if isinstance(file, list):
        moving_path = file[0]
        additional_files = file[1:]
    else:
        moving_path = file
        additional_files = None

    fixed_path = '/usr/share/fsl/5.0/data/standard/MNI152_T1_1mm_brain.nii.gz'

    fixed = ants.image_read(fixed_path)
    moving = ants.image_read(moving_path)

    if flag:
        pat_id = moving_path.split(os.path.sep)[-2]
        out_dir = os.path.join(out_base, pat_id)
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = out_base

    file_name = os.path.basename(moving_path)
    mni_file_path = os.path.join(out_dir, f'{prefix}_{file_name}')
    mat_path = os.path.join(out_dir, 'Affine_MNI.mat')

    print(f'Registering {moving_path} to MNI...')
    mytx = ants.registration(fixed=fixed, moving=moving, type_of_transform='SyN')
    print('Done!')
    ants.image_write(mytx['warpedmovout'], mni_file_path)
    shutil.copyfile(mytx['fwdtransforms'][-1], mat_path)

    if additional_files is not None:
        for f in additional_files:
            moving = ants.image_read(f)
            file_name = os.path.basename(f)
            mni_file_path = os.path.join(out_dir, f'{prefix}_{file_name}')
            print(f'Registering {f} to MNI...')
            if file_name == 'VessTerritory_f3d.nii.gz':
                warped_img = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=mytx['fwdtransforms'],
                                                   interpolator="genericLabel")
            else:
                warped_img = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=mytx['fwdtransforms'],
                                                   interpolator="linear")
            ants.image_write(warped_img, mni_file_path)


def augment_symmetric_modality(file_path, prefix='sym'):
    filename_in = os.path.basename(file_path)
    path_out = os.path.dirname(file_path)
    flipped_ct_path = os.path.join(path_out, 'flipped_{}'.format(filename_in))
    sym_ct_path = os.path.join(path_out, '{}_{}'.format(prefix, filename_in))
    mat_path = os.path.join(path_out, 'Affine_MNI.mat')

    original_nib = nib.load(file_path)
    original_data = original_nib.get_data()

    # Flip and save "flipped image"
    flipped_data = np.flip(original_data, axis=0)
    img = nib.Nifti1Image(flipped_data, original_nib.affine, original_nib.header)
    nib.save(img, flipped_ct_path)

    fixed = ants.image_read(file_path)
    moving = ants.image_read(flipped_ct_path)


    print(f'Registering {flipped_ct_path} to original...')
    mytx = ants.registration(fixed=fixed, moving=moving, type_of_transform='SyN')
    print('Done!')
    ants.image_write(mytx['warpedmovout'], sym_ct_path)
    shutil.copyfile(mytx['fwdtransforms'][-1], mat_path)
   

    # Delete flipped
    delete_command_template = 'rm {}'
    subprocess.check_output(['bash', '-c', delete_command_template.format(flipped_ct_path)])

   