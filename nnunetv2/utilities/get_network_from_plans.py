from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn

import os
import torch
import torch.nn.functional as F
import numpy as np
# from nnunetv2.network_architecture.Coformer import Coformer3D
# from nnunetv2.network_architecture.Coformer import UNetDecoder as Decoder
# from nnunetv2.network_architecture.Coformer import nnUNetDecoder
from torch.nn.init import trunc_normal_

# pycharm 不需要，但是Linux终端需要下面两句
########################################################
import sys
sys.path.append("/home/tanxz/Codes/nnUNet")
#######################################################

from Three_d.clseg import ClSeg
# from nnunetv2.network_architecture.ClSeg import ClSeg
# from nnunetv2.network_architecture.TransHRNet.TransHRNet_S import TransHRNet_v2
# from nnunetv2.network_architecture.ISNet import ISNet
# from nnunetv2.network_architecture.h2former import H2Former
# from nnunetv2.network_architecture.batformer import BATFormer
# from nnunetv2.network_architecture.models import FAT_Net
# from nnunetv2.network_architecture.hiformer import HiFormer
# from nnunetv2.network_architecture.models import SAN
# from nnunetv2.network_architecture.lambdaunet import LambdaUNet
# from nnunetv2.network_architecture.models import SUNETx5
# from nnunetv2.network_architecture.models import UNetAM, UNetGC, UNetAttn
# from nnunetv2.network_architecture.models import ADHDC_Net, LCOVNet
# from nnunetv2.network_architecture.models import DFormer
# from nnunetv2.network_architecture.phtrans import PHTrans
# from nnunetv2.network_architecture.factorizer import SegmentationFactorizer, SWMatricize, FactorizerSubblock, NMF
# from nnunetv2.network_architecture.SlimUNETR import SlimUNETR


class InitWeights_He(object):

    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or \
                isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)


class LWCTrans(nn.Module):
    def __init__(self, in_channels=1, num_classes=None,
                 conv_kernel_size=None, pool_op_kernel_sizes=None,
                 base_num_features=32, max_num_features=320, ):
        super(LWCTrans, self).__init__()

        self.MODEL_NUM_CLASSES = num_classes

        embed_dims = [min(base_num_features * 2 ** i, max_num_features) for i in range(len(conv_kernel_size))]
        output_features = base_num_features
        stride = pool_op_kernel_sizes
        padding = [[1 if i == 3 else 0 for i in krnl] for krnl in conv_kernel_size]

        self.backbone = Coformer3D(in_chans=in_channels, kernel_size=conv_kernel_size, stride=stride, padding=padding,
                                   embed_dims=embed_dims)

        self.decoder = Decoder(num_class=num_classes, embed_dims=embed_dims[::-1],
                               kernel_size=conv_kernel_size[:0:-1],
                               stride=stride[:0:-1], padding=padding[:0:-1])
        backbone = sum([param.nelement() for param in self.backbone.parameters()])
        total = sum([param.nelement() for param in self.parameters()])
        print('  + Number of Backbone Params: %.2f(e6) M' % (backbone / 1e6))
        print('  + Number of Total Params: %.2f(e6) M' % (total / 1e6))
        self.apply(InitWeights_He())

    def forward(self, inputs):
        # # %%%%%%%%%%%%% LWCTrans
        B, C, D, H, W = inputs.shape
        x = inputs
        x = self.backbone(x)
        x = self.decoder(x)
        return x


class SegmentationNetwork(nn.Module):
    """
    All Segmentation Networks
    """

    def __init__(self, in_channels=1, num_classes=None, img_size=None,
                 conv_kernel_size=None, pool_op_kernel_sizes=None, base_num_features=32, max_num_features=320,
                 deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        # self.network = LWCTrans(in_channels=in_channels, num_classes=num_classes,
        #                         conv_kernel_size=conv_kernel_size,
        #                         pool_op_kernel_sizes=pool_op_kernel_sizes,
        #                         base_num_features=base_num_features, max_num_features=max_num_features)

        self.network = ClSeg(in_channels, 16, num_classes,
                             len(pool_op_kernel_sizes),
                             2, 2, nn.Conv3d, nn.InstanceNorm3d, {'eps': 1e-5, 'affine': True}, nn.Dropout3d,
                             {'p': 0, 'inplace': True},
                             nn.LeakyReLU, {'negative_slope': 1e-2, 'inplace': True}, True, False, lambda x: x,
                             InitWeights_He(1e-2),
                             pool_op_kernel_sizes, conv_kernel_size, False, True,
                             True)
        # self.network = PHTrans(img_size=img_size,
        #                        base_num_features=24,
        #                        num_classes=num_classes,
        #                        num_pool=len(pool_op_kernel_sizes[-3:]),
        #                        image_channels=in_channels,
        #                        pool_op_kernel_sizes=pool_op_kernel_sizes[-4:],
        #                        conv_kernel_sizes=conv_kernel_size[-6:],
        #                        deep_supervision=False,
        #                        max_num_features=24 * 13,
        #                        depths=[2, 2, 2, 2],
        #                        num_heads=[3, 6, 12, 24],
        #                        window_size=[2, 2, 2],
        #                        drop_path_rate=0.2)
        # self.network = TransHRNet_v2(num_classes=num_classes, deep_supervision=False)
        # self.network = H2Former(image_size=img_size, num_class=num_classes)
        # self.network = BATFormer(n_channels=in_channels, n_classes=num_classes, imgsize=img_size[0])
        # self.network = FAT_Net(n_channels=in_channels, n_classes=num_classes)
        # self.network = HiFormer(img_size=img_size[0], in_chans=in_channels, n_classes=num_classes)
        # self.network = LambdaUNet(n_channels=in_channels, n_classes=num_classes)
        # self.network = ISNet(in_channel=in_channels, num_classes=num_classes)
        # self.network = UNetAM(in_channels, num_classes)
        # self.network = UNetGC(in_channels, num_classes)
        # self.network = SUNETx5(in_ch=in_channels, out_ch=num_classes)
        # self.network = UNetAttn(in_channel=in_channels, num_classes=num_classes)
        # self.network = LCOVNet(in_channels, num_classes)
        # self.network = ADHDC_Net(4, 4, 32)
        # self.network = DFormer(in_chan=in_channels, num_classes=num_classes, deep_supervision=False)
        # self.network = SlimUNETR(in_channels=in_channels, out_channels=num_classes, embedding_dim=100)

    def forward(self, x):
        seg_output = self.network(x)
        if self.deep_supervision:
            if not isinstance(seg_output, list) and not isinstance(seg_output, tuple):
                return [seg_output]
            else:
                return seg_output
        else:
            if not isinstance(seg_output, list) and not isinstance(seg_output, tuple):
                return seg_output
            else:
                return seg_output[0]

def get_network_from_plans(plans_manager: PlansManager,
                           dataset_json: dict,
                           configuration_manager: ConfigurationManager,
                           num_input_channels: int,
                           deep_supervision: bool = True):
    """
    we may have to change this in the future to accommodate other plans -> network mappings

    num_input_channels can differ depending on whether we do cascade. Its best to make this info available in the
    trainer rather than inferring it again from the plans here.
    """
    num_stages = len(configuration_manager.conv_kernel_sizes)

    dim = len(configuration_manager.conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)

    label_manager = plans_manager.get_label_manager(dataset_json)

    segmentation_network_class_name = configuration_manager.UNet_class_name
    mapping = {
        'PlainConvUNet': PlainConvUNet,
        'ResidualEncoderUNet': ResidualEncoderUNet
    }
    kwargs = {
        'PlainConvUNet': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        },
        'ResidualEncoderUNet': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        }
    }
    assert segmentation_network_class_name in mapping.keys(), 'The network architecture specified by the plans file ' \
                                                              'is non-standard (maybe your own?). Yo\'ll have to dive ' \
                                                              'into either this ' \
                                                              'function (get_network_from_plans) or ' \
                                                              'the init of your nnUNetModule to accommodate that.'
    network_class = mapping[segmentation_network_class_name]

    conv_or_blocks_per_stage = {
        'n_conv_per_stage'
        if network_class != ResidualEncoderUNet else 'n_blocks_per_stage': configuration_manager.n_conv_per_stage_encoder,
        'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
    }
    # network class name!!
    model = network_class(
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
                                configuration_manager.unet_max_num_features) for i in range(num_stages)],
        conv_op=conv_op,
        kernel_sizes=configuration_manager.conv_kernel_sizes,
        strides=configuration_manager.pool_op_kernel_sizes,
        num_classes=label_manager.num_segmentation_heads,
        deep_supervision=deep_supervision,
        **conv_or_blocks_per_stage,
        **kwargs[segmentation_network_class_name]
    )
    model.apply(InitWeights_He(1e-2))
    if network_class == ResidualEncoderUNet:
        model.apply(init_last_bn_before_add_to_0)
    return model

##
def get_network_from_plans_customized(plans_manager: PlansManager,
                           dataset_json: dict,
                           configuration_manager: ConfigurationManager,
                           num_input_channels: int,
                           deep_supervision: bool = True):
    """
    we may have to change this in the future to accommodate other plans -> network mappings

    num_input_channels can differ depending on whether we do cascade. Its best to make this info available in the
    trainer rather than inferring it again from the plans here.
    """
    num_stages = len(configuration_manager.conv_kernel_sizes)

    dim = len(configuration_manager.conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)

    label_manager = plans_manager.get_label_manager(dataset_json)

    in_channels = num_input_channels
    num_classes = label_manager.num_segmentation_heads
    img_size = configuration_manager.patch_size
    # conv_kernel_size = configuration_manager.conv_kernel_sizes
    # pool_op_kernel_sizes = configuration_manager.pool_op_kernel_sizes
    ############ 师兄的配置文件 ################
    conv_kernel_size = [[1, 3, 3], [1, 3, 3], [1, 3, 3], [1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
    pool_op_kernel_sizes = [[1, 2, 2], [1, 2, 2], [1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2]]
    base_num_features = 8 if len(conv_kernel_size) >= 7 else 16
    max_num_features = configuration_manager.unet_max_num_features
    ######### 师兄的配置文件 #####################

    model = SegmentationNetwork(in_channels=in_channels, num_classes=num_classes, img_size=img_size,
                                conv_kernel_size=conv_kernel_size, pool_op_kernel_sizes=pool_op_kernel_sizes,
                                base_num_features=base_num_features, max_num_features=max_num_features,
                                deep_supervision=deep_supervision)
    return model