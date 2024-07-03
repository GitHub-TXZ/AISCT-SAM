# AISCT-SAM
The code for the paper "AISCT-SAM: Customized SAM-Med2D with 3D Context Awareness and Self-Prompt Generation for Fully Automatic Acute Ischemic Stroke Lesion Segmentation on NonContrast CT Scans" submitted to IEEE TMI. <br />


## Requirements
CUDA 11.7<br />
Python 3.10.13<br /> 
Pytorch 2.0.0<br />
Torchvision 0.15.0<br />
batchgenerators 0.25<br />
SimpleITK 2.3.0 <br />
scipy 1.11.3 <br />

## Usage

### 0. Installation
* Install our modified nnUNet as below
  
```
git clone https://github.com/GitHub-TXZ/AISCT-SAM.git
cd AISCT-SAM
pip install -e .

```
### 1 Acute Ischemic Stroke Dataset (AISD)
### 1.1 Dataset access
AISD dataset can be downloaded from (https://github.com/griffinliang/aisd). 
### 1.2 Skull-stripping
After converting the DICOM files of the AISD dataset to NIfTI format, perform skull stripping according to the instructions at https://github.com/WuChanada/StripSkullCT.
### 1.3 Flip-Registration
Then, perform flip registration according to ./myscripts/Registration. Finally, organize the dataset in the nnUNet-expected format according to the code in nnUNet/nnunet/dataset_conversion.
### 1.3 Pre-processing
Some compared methods use the same pre-processing steps as nnUNet. The documentation of the pre-processing can be found at [[DOC]](./nnUNet/documentation) <br />

### 1.4 Training
conda activate <YOUR ENV NAME>
Simply run the following in your command line:
* Run `CUDA_VISIBLE_DEVICES=0 nnUNetv2_train -dataset_name_or_id TASK_ID -model_name AIS_SAM -ex_name Ex1@b_2_p_20_256_256_s_3.0_0.4375_0.4375` for training.

### 1.5 Testing 
* Run `CUDA_VISIBLE_DEVICES=0 nnUNetv2_train -dataset_name_or_id TASK_ID -model_name AIS_SAM -ex_name Ex1@b_2_p_20_256_256_s_3.0_0.4375_0.4375 --val` for testing.

### 2.1 Pre-trained model
The pre-trained model of AISD dataset can be downloaded from [[Baidu YUN]](https://pan.baidu.com/s/1RmswEZsQewr7UcC14UCKMA) with the password "4phx".

### 2.2 Reproduction details and codes 
During reproduction, for the methods (e.g. LambdaUNet [1], UNet-AM [2], UNet-GC [3]) that do not publish their codes, we endeavored to implement their approaches by following
the technical details provided in their papers. our reproduced codes of these methods can be found at [[DOC]](./ClSeg_package/ClSeg/network_architecture) and [[DOC]](./ClSeg_package/ClSeg/network_architecture/models)

For the compared methods with Open-source codes, we directly use their codes for AIS lesion segmentation on 2 AIS datasets. The links of their open-source codes can are listed as follows: <br />

[[AttnUnet2D]](https://github.com/sfczekalski/attention_unet) </br>
[[Swin-Unet]](https://github.com/HuCaoFighting/Swin-Unet) </br>
[[TransUNet]](https://github.com/Beckschen/TransUNet) </br>
[[FAT-Net]](https://github.com/SZUcsh/FAT-Net) </br>
[[AttnUNet3D]](https://github.com/mobarakol/3D_Attention_UNet) </br>
[[nnFormer]](https://github.com/282857341/nnFormer) </br>
[[UNETR]](https://github.com/282857341/nnFormer) </br>
[[CoTr]](https://github.com/YtongXie/CoTr) </br>
[[nnUNet]](https://github.com/MIC-DKFZ/nnUNet) </br>
[[UNet-RF]](https://github.com/WuChanada/Acute-ischemic-lesion-segmentation-in-NCCT)

Note that for all compared methods, to perform fair comparisons, we use the same pre-processing steps (as 1.2 Pre-prcoessing) and the same data split. and in the paper all compared results were derived from our reproduction experiments.


## Acknowledgements
Part of codes are reused from the nnU-Net. Thanks to Fabian Isensee for the codes of nnU-Net.
