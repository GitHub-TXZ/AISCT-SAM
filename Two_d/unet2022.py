from einops import rearrange
from copy import deepcopy
# from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
import torch
import numpy as np
# from nnunet.network_architecture.initialization import InitWeights_He
# from nnunet.network_architecture.neural_network import SegmentationNetwork
import torch.nn.functional


import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_3tuple, trunc_normal_

import ml_collections

Synapse_train_list = ['label0006', 'label0007', 'label0009', 'label0010', 'label0021', 'label0023', 'label0024',
                      'label0026', 'label0027', 'label0031', 'label0033', 'label0034', 'label0039', 'label0040',
                      'label0005', 'label0028', 'label0030', 'label0037']
Synapse_val_list = ['label0001', 'label0002', 'label0003', 'label0004', 'label0008', 'label0022', 'label0025',
                    'label0029', 'label0032', 'label0035', 'label0036', 'label0038']

ACDC_train_list = ['patient001_frame01', 'patient001_frame12', 'patient004_frame01', 'patient004_frame15',
                   'patient005_frame01', 'patient005_frame13', 'patient006_frame01', 'patient006_frame16',
                   'patient007_frame01', 'patient007_frame07', 'patient010_frame01', 'patient010_frame13',
                   'patient011_frame01', 'patient011_frame08', 'patient013_frame01', 'patient013_frame14',
                   'patient015_frame01', 'patient015_frame10', 'patient016_frame01', 'patient016_frame12',
                   'patient018_frame01', 'patient018_frame10', 'patient019_frame01', 'patient019_frame11',
                   'patient020_frame01', 'patient020_frame11', 'patient021_frame01', 'patient021_frame13',
                   'patient022_frame01', 'patient022_frame11', 'patient023_frame01', 'patient023_frame09',
                   'patient025_frame01', 'patient025_frame09', 'patient026_frame01', 'patient026_frame12',
                   'patient027_frame01', 'patient027_frame11', 'patient028_frame01', 'patient028_frame09',
                   'patient029_frame01', 'patient029_frame12', 'patient030_frame01', 'patient030_frame12',
                   'patient031_frame01', 'patient031_frame10', 'patient032_frame01', 'patient032_frame12',
                   'patient033_frame01', 'patient033_frame14', 'patient034_frame01', 'patient034_frame16',
                   'patient035_frame01', 'patient035_frame11', 'patient036_frame01', 'patient036_frame12',
                   'patient037_frame01', 'patient037_frame12', 'patient038_frame01', 'patient038_frame11',
                   'patient039_frame01', 'patient039_frame10', 'patient040_frame01', 'patient040_frame13',
                   'patient041_frame01', 'patient041_frame11', 'patient043_frame01', 'patient043_frame07',
                   'patient044_frame01', 'patient044_frame11', 'patient045_frame01', 'patient045_frame13',
                   'patient046_frame01', 'patient046_frame10', 'patient047_frame01', 'patient047_frame09',
                   'patient050_frame01', 'patient050_frame12', 'patient051_frame01', 'patient051_frame11',
                   'patient052_frame01', 'patient052_frame09', 'patient054_frame01', 'patient054_frame12',
                   'patient056_frame01', 'patient056_frame12', 'patient057_frame01', 'patient057_frame09',
                   'patient058_frame01', 'patient058_frame14', 'patient059_frame01', 'patient059_frame09',
                   'patient060_frame01', 'patient060_frame14', 'patient061_frame01', 'patient061_frame10',
                   'patient062_frame01', 'patient062_frame09', 'patient063_frame01', 'patient063_frame16',
                   'patient065_frame01', 'patient065_frame14', 'patient066_frame01', 'patient066_frame11',
                   'patient068_frame01', 'patient068_frame12', 'patient069_frame01', 'patient069_frame12',
                   'patient070_frame01', 'patient070_frame10', 'patient071_frame01', 'patient071_frame09',
                   'patient072_frame01', 'patient072_frame11', 'patient073_frame01', 'patient073_frame10',
                   'patient074_frame01', 'patient074_frame12', 'patient075_frame01', 'patient075_frame06',
                   'patient076_frame01', 'patient076_frame12', 'patient077_frame01', 'patient077_frame09',
                   'patient078_frame01', 'patient078_frame09', 'patient080_frame01', 'patient080_frame10',
                   'patient082_frame01', 'patient082_frame07', 'patient083_frame01', 'patient083_frame08',
                   'patient084_frame01', 'patient084_frame10', 'patient085_frame01', 'patient085_frame09',
                   'patient086_frame01', 'patient086_frame08', 'patient087_frame01', 'patient087_frame10']
ACDC_val_list = ['patient089_frame01', 'patient089_frame10', 'patient090_frame04', 'patient090_frame11',
                 'patient091_frame01', 'patient091_frame09', 'patient093_frame01', 'patient093_frame14',
                 'patient094_frame01', 'patient094_frame07', 'patient096_frame01', 'patient096_frame08',
                 'patient097_frame01', 'patient097_frame11', 'patient098_frame01', 'patient098_frame09',
                 'patient099_frame01', 'patient099_frame09', 'patient100_frame01', 'patient100_frame13']

EM_train_list = ['train-labels00', 'train-labels01', 'train-labels02', 'train-labels03', 'train-labels04',
                 'train-labels05', 'train-labels06', 'train-labels07', 'train-labels08', 'train-labels10',
                 'train-labels12', 'train-labels14', 'train-labels15', 'train-labels16', 'train-labels17',
                 'train-labels19', 'train-labels20', 'train-labels23', 'train-labels24', 'train-labels25',
                 'train-labels26', 'train-labels27', 'train-labels28', 'train-labels29']
EM_val_list = ['train-labels09', 'train-labels11', 'train-labels13']

ISIC_train_list = ['0000001', '0000002', '0000007', '0000008', '0000009', '0000010', '0000011', '0000017', '0000018',
                   '0000019', '0000021', '0000024', '0000025', '0000026', '0000028', '0000029', '0000030', '0000031',
                   '0000032', '0000034', '0000035', '0000038', '0000039', '0000041', '0000042', '0000044', '0000046',
                   '0000047', '0000049', '0000050', '0000051', '0000054', '0000055', '0000058', '0000059', '0000061',
                   '0000062', '0000063', '0000065', '0000067', '0000068', '0000073', '0000074', '0000075', '0000077',
                   '0000078', '0000085', '0000086', '0000087', '0000091', '0000093', '0000094', '0000095', '0000096',
                   '0000097', '0000103', '0000104', '0000105', '0000108', '0000110', '0000112', '0000114', '0000116',
                   '0000118', '0000119', '0000120', '0000121', '0000123', '0000124', '0000127', '0000128', '0000131',
                   '0000133', '0000134', '0000135', '0000137', '0000139', '0000140', '0000142', '0000143', '0000145',
                   '0000146', '0000148', '0000150', '0000151', '0000152', '0000154', '0000156', '0000157', '0000162',
                   '0000166', '0000167', '0000170', '0000171', '0000173', '0000176', '0000181', '0000182', '0000183',
                   '0000185', '0000186', '0000187', '0000190', '0000191', '0000193', '0000199', '0000204', '0000207',
                   '0000208', '0000209', '0000210', '0000211', '0000214', '0000215', '0000217', '0000218', '0000219',
                   '0000220', '0000224', '0000225', '0000232', '0000235', '0000236', '0000237', '0000239', '0000240',
                   '0000242', '0000243', '0000244', '0000245', '0000249', '0000250', '0000251', '0000255', '0000256',
                   '0000259', '0000260', '0000262', '0000263', '0000264', '0000265', '0000268', '0000269', '0000274',
                   '0000276', '0000277', '0000278', '0000280', '0000285', '0000288', '0000290', '0000293', '0000294',
                   '0000307', '0000313', '0000314', '0000317', '0000321', '0000323', '0000324', '0000326', '0000329',
                   '0000330', '0000331', '0000332', '0000333', '0000337', '0000338', '0000339', '0000341', '0000345',
                   '0000346', '0000347', '0000348', '0000349', '0000350', '0000351', '0000352', '0000353', '0000355',
                   '0000358', '0000359', '0000360', '0000361', '0000363', '0000365', '0000366', '0000369', '0000370',
                   '0000374', '0000376', '0000379', '0000381', '0000382', '0000383', '0000384', '0000385', '0000386',
                   '0000390', '0000391', '0000395', '0000397', '0000403', '0000408', '0000409', '0000410', '0000412',
                   '0000413', '0000416', '0000419', '0000421', '0000425', '0000426', '0000427', '0000431', '0000434',
                   '0000436', '0000439', '0000442', '0000443', '0000445', '0000447', '0000451', '0000453', '0000454',
                   '0000455', '0000457', '0000458', '0000460', '0000461', '0000463', '0000465', '0000467', '0000468',
                   '0000469', '0000471', '0000474', '0000477', '0000478', '0000480', '0000483', '0000485', '0000486',
                   '0000489', '0000491', '0000492', '0000493', '0000495', '0000496', '0000498', '0000500', '0000503',
                   '0000504', '0000505', '0000506', '0000507', '0000513', '0000514', '0000516', '0000521', '0000522',
                   '0000523', '0000528', '0000529', '0000530', '0000531', '0000532', '0000535', '0000536', '0000538',
                   '0000541', '0000542', '0000543', '0000544', '0000545', '0000546', '0000551', '0000552', '0000555',
                   '0000556', '0000882', '0000900', '0000999', '0001102', '0001105', '0001118', '0001119', '0001126',
                   '0001133', '0001134', '0001140', '0001148', '0001152', '0001163', '0001184', '0001187', '0001188',
                   '0001191', '0001212', '0001213', '0001216', '0001247', '0001267', '0001275', '0001296', '0001306',
                   '0001374', '0001385', '0001442', '0002093', '0002206', '0002251', '0002287', '0002353', '0002374',
                   '0002438', '0002453', '0002459', '0002469', '0002476', '0002489', '0002616', '0002647', '0002780',
                   '0002806', '0002836', '0002879', '0002885', '0002975', '0002976', '0003051', '0003174', '0003308',
                   '0003346', '0004110', '0004166', '0004309', '0004715', '0005187', '0005247', '0005548', '0005564',
                   '0005620', '0005639', '0006021', '0006114', '0006326', '0006350', '0006776', '0006800', '0006940',
                   '0006982', '0007038', '0007475', '0007760', '0007788', '0008145', '0008256', '0008280', '0008294',
                   '0008347', '0008396', '0008403', '0008524', '0008541', '0008552', '0008807', '0008879', '0008913',
                   '0008993', '0009160', '0009165', '0009188', '0009252', '0009297', '0009344', '0009430', '0009504',
                   '0009505', '0009533', '0009583', '0009758', '0009800', '0009868', '0009870', '0009871', '0009873',
                   '0009875', '0009877', '0009883', '0009884', '0009888', '0009893', '0009895', '0009896', '0009897',
                   '0009899', '0009900', '0009904', '0009910', '0009911', '0009912', '0009914', '0009919', '0009921',
                   '0009929', '0009933', '0009936', '0009937', '0009938', '0009939', '0009940', '0009944', '0009947',
                   '0009949', '0009950', '0009951', '0009953', '0009961', '0009962', '0009963', '0009964', '0009966',
                   '0009967', '0009968', '0009969', '0009972', '0009973', '0009974', '0009976', '0009979', '0009983',
                   '0009986', '0009987', '0009991', '0009995', '0010000', '0010001', '0010002', '0010003', '0010005',
                   '0010010', '0010014', '0010015', '0010017', '0010019', '0010021', '0010022', '0010024', '0010025',
                   '0010032', '0010035', '0010036', '0010040', '0010042', '0010043', '0010046', '0010051', '0010052',
                   '0010053', '0010054', '0010056', '0010057', '0010060', '0010063', '0010065', '0010067', '0010069',
                   '0010070', '0010071', '0010075', '0010078', '0010079', '0010080', '0010081', '0010083', '0010086',
                   '0010090', '0010093', '0010094', '0010102', '0010104', '0010168', '0010169', '0010174', '0010176',
                   '0010177', '0010178', '0010182', '0010185', '0010189', '0010191', '0010194', '0010212', '0010213',
                   '0010218', '0010219', '0010220', '0010222', '0010223', '0010225', '0010226', '0010227', '0010230',
                   '0010232', '0010233', '0010235', '0010236', '0010237', '0010239', '0010240', '0010241', '0010244',
                   '0010246', '0010247', '0010248', '0010249', '0010252', '0010256', '0010263', '0010264', '0010265',
                   '0010267', '0010320', '0010321', '0010322', '0010323', '0010324', '0010325', '0010327', '0010329',
                   '0010332', '0010333', '0010334', '0010335', '0010339', '0010344', '0010349', '0010350', '0010351',
                   '0010352', '0010356', '0010357', '0010358', '0010361', '0010362', '0010365', '0010367', '0010370',
                   '0010371', '0010380', '0010382', '0010435', '0010436', '0010438', '0010440', '0010441', '0010442',
                   '0010443', '0010445', '0010450', '0010455', '0010457', '0010458', '0010459', '0010461', '0010462',
                   '0010465', '0010466', '0010468', '0010471', '0010472', '0010473', '0010475', '0010476', '0010479',
                   '0010480', '0010481', '0010487', '0010488', '0010490', '0010491', '0010492', '0010493', '0010496',
                   '0010497', '0010554', '0010557', '0010562', '0010566', '0010567', '0010568', '0010569', '0010570',
                   '0010571', '0010573', '0010575', '0010576', '0010577', '0010585', '0010586', '0010589', '0010593',
                   '0010594', '0010595', '0010602', '0010603', '0010605', '0010844', '0010848', '0010849', '0010850',
                   '0010851', '0010852', '0010853', '0010857', '0010858', '0010860', '0010861', '0010862', '0011079',
                   '0011084', '0011085', '0011088', '0011095', '0011097', '0011099', '0011100', '0011109', '0011114',
                   '0011115', '0011116', '0011117', '0011118', '0011120', '0011121', '0011123', '0011124', '0011126',
                   '0011127', '0011128', '0011130', '0011131', '0011135', '0011137', '0011139', '0011140', '0011141',
                   '0011145', '0011146', '0011159', '0011161', '0011163', '0011164', '0011165', '0011166', '0011169',
                   '0011170', '0011200', '0011202', '0011203', '0011207', '0011208', '0011210', '0011211', '0011212',
                   '0011214', '0011215', '0011217', '0011218', '0011223', '0011225', '0011226', '0011228', '0011230',
                   '0011295', '0011296', '0011297', '0011299', '0011301', '0011303', '0011306', '0011315', '0011317',
                   '0011323', '0011324', '0011326', '0011327', '0011328', '0011329', '0011330', '0011331', '0011332',
                   '0011334', '0011341', '0011343', '0011345', '0011346', '0011347', '0011348', '0011352', '0011353',
                   '0011354', '0011356', '0011357', '0011358', '0011360', '0011361', '0011362', '0011372', '0011378',
                   '0011380', '0011382', '0011383', '0011385', '0011387', '0011390', '0011397', '0011400', '0011402']
ISIC_train_list2 = ['0000000', '0000004', '0000006', '0000016', '0000045', '0000048', '0000060', '0000079', '0000080',
                    '0000081', '0000082', '0000089', '0000100', '0000102', '0000109', '0000122', '0000130', '0000147',
                    '0000153', '0000155', '0000159', '0000163', '0000175', '0000179', '0000184', '0000189', '0000192',
                    '0000203', '0000205', '0000206', '0000216', '0000221', '0000223', '0000229', '0000247', '0000252',
                    '0000261', '0000275', '0000281', '0000282', '0000283', '0000292', '0000295', '0000297', '0000300',
                    '0000301', '0000303', '0000315', '0000316', '0000322', '0000336', '0000342', '0000344', '0000364',
                    '0000367', '0000372', '0000387', '0000396', '0000415', '0000423', '0000444', '0000452', '0000473',
                    '0000475', '0000481', '0000488', '0000511', '0000517', '0000519', '0000520', '0000524', '0000548',
                    '0000554', '0001106', '0001254', '0001262', '0001286', '0001292', '0001367', '0001372', '0001423',
                    '0001449', '0001742', '0002439', '0002488', '0002948', '0003005', '0004168', '0004985', '0005555',
                    '0005666', '0005787', '0006612', '0006795', '0007087', '0007557', '0008029', '0008236', '0008528',
                    '0008785', '0009599', '0009860', '0009905', '0009909', '0009915', '0009917', '0009925', '0009932',
                    '0009934', '0009935', '0009941', '0009942', '0009960', '0009971', '0009975', '0009978', '0009981',
                    '0010006', '0010029', '0010044', '0010064', '0010066', '0010068', '0010074', '0010087', '0010091',
                    '0010101', '0010105', '0010170', '0010184', '0010186', '0010203', '0010204', '0010205', '0010228',
                    '0010242', '0010251', '0010262', '0010266', '0010317', '0010318', '0010319', '0010330', '0010337',
                    '0010341', '0010342', '0010364', '0010372', '0010439', '0010447', '0010464', '0010467', '0010495',
                    '0010558', '0010572', '0010581', '0010590', '0010864', '0011082', '0011102', '0011105', '0011119',
                    '0011125', '0011136', '0011144', '0011156', '0011157', '0011158', '0011173', '0011199', '0011220',
                    '0011229', '0011304', '0011322', '0011339', '0011350', '0011366', '0011373', '0011393', '0011398']
ISIC_train_list.extend(ISIC_train_list2)
ISIC_val_list = []


def EM_512():
    config = ml_collections.ConfigDict()

    config.pretrain = True
    config.deep_supervision = True
    config.train_list = EM_train_list
    config.val_list = EM_val_list

    config.hyper_parameter = ml_collections.ConfigDict()
    config.hyper_parameter.crop_size = [512, 512]
    config.hyper_parameter.batch_size = 8
    config.hyper_parameter.base_learning_rate = 7e-4
    config.hyper_parameter.model_size = 'Tiny'
    config.hyper_parameter.blocks_num = [3, 3, 3, 3]
    config.hyper_parameter.window_size = [16, 16, 16, 8]
    config.hyper_parameter.val_eval_criterion_alpha = 0.9
    config.hyper_parameter.epochs_num = 2000
    config.hyper_parameter.convolution_stem_down = 8

    return config


def ISIC_512():
    config = ml_collections.ConfigDict()

    config.pretrain = True
    config.deep_supervision = True
    config.train_list = ISIC_train_list
    config.val_list = ISIC_val_list

    config.hyper_parameter = ml_collections.ConfigDict()
    config.hyper_parameter.crop_size = [512, 512]
    config.hyper_parameter.batch_size = 16
    config.hyper_parameter.base_learning_rate = 1.3e-4
    config.hyper_parameter.model_size = 'Tiny'
    config.hyper_parameter.blocks_num = [3, 3, 3, 3]
    config.hyper_parameter.window_size = [8, 8, 16, 8]
    config.hyper_parameter.val_eval_criterion_alpha = 0
    config.hyper_parameter.epochs_num = 75
    config.hyper_parameter.convolution_stem_down = 8

    return config


def ACDC_224():
    config = ml_collections.ConfigDict()

    config.pretrain = True
    config.deep_supervision = False
    config.train_list = ACDC_train_list
    config.val_list = ACDC_val_list

    config.hyper_parameter = ml_collections.ConfigDict()
    config.hyper_parameter.crop_size = [224, 224]
    config.hyper_parameter.batch_size = 8
    config.hyper_parameter.base_learning_rate = 1e-4
    config.hyper_parameter.model_size = 'Large'
    config.hyper_parameter.blocks_num = [3, 3, 3, 3]
    config.hyper_parameter.window_size = [7, 7, 14, 7]
    config.hyper_parameter.val_eval_criterion_alpha = 0.9
    config.hyper_parameter.epochs_num = 500
    config.hyper_parameter.convolution_stem_down = 4

    return config


def Synapse_224():
    config = ml_collections.ConfigDict()

    config.pretrain = True
    config.deep_supervision = True
    config.train_list = Synapse_train_list
    config.val_list = Synapse_val_list

    config.hyper_parameter = ml_collections.ConfigDict()
    config.hyper_parameter.crop_size = [224, 224]
    config.hyper_parameter.batch_size = 16
    config.hyper_parameter.base_learning_rate = 1e-4
    config.hyper_parameter.model_size = 'Base'
    config.hyper_parameter.blocks_num = [3, 3, 3, 3]
    config.hyper_parameter.window_size = [7, 7, 14, 7]
    config.hyper_parameter.val_eval_criterion_alpha = 0.
    config.hyper_parameter.epochs_num = 2700
    config.hyper_parameter.convolution_stem_down = 4

    return config


def Synapse_320():
    config = ml_collections.ConfigDict()

    config.pretrain = True
    config.deep_supervision = True
    config.train_list = Synapse_train_list
    config.val_list = Synapse_val_list

    config.hyper_parameter = ml_collections.ConfigDict()
    config.hyper_parameter.crop_size = [320, 320]
    config.hyper_parameter.batch_size = 8
    config.hyper_parameter.base_learning_rate = 1e-4
    config.hyper_parameter.model_size = 'Tiny'
    config.hyper_parameter.blocks_num = [3, 3, 3, 3]
    config.hyper_parameter.window_size = [10, 10, 20, 10]
    config.hyper_parameter.val_eval_criterion_alpha = 0.
    config.hyper_parameter.epochs_num = 1300
    config.hyper_parameter.convolution_stem_down = 4

    return config


CONFIGS = {
    'EM_512': EM_512(),
    'ISIC_512': ISIC_512(),
    'ACDC_224': ACDC_224(),
    'Synapse_224': Synapse_224(),
    'Synapse_320': Synapse_320(),
}


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B,  H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_s = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_s, coords_h]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1

        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None,dw=None):
        B_, N, C = x.shape

        qkv = self.qkv(x)

        qkv=qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        if dw is not None:
            x = x + dw
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MSABlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)

        self.attn = WindowAttention(
            dim, window_size=to_3tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.dwconv=nn.Conv2d(dim,dim,kernel_size=7,padding=3,groups=dim)
    def forward(self, x, mask_matrix):

        B, H,W, C = x.shape

        assert H * W==self.input_resolution[0]*self.input_resolution[1], "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size

        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size,-self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        dw=shifted_x.permute(0,3,1,2).contiguous()
        dw = self.dwconv(dw)
        dw = dw.permute(0,2,3,1).contiguous()
        dw = window_partition(dw, self.window_size)  # nW*B, window_size, window_size, C
        dw = dw.view(-1, self.window_size * self.window_size,
                                   C)

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size,
                                   C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask,dw=dw)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0 :
            x = x[:, :H, :W, :].contiguous()

        #x = x.view(B,  H * W, C)
        x = shortcut + self.drop_path(x)

        return x

class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(dim,dim*2,kernel_size=3,stride=2,padding=1)
        self.norm = norm_layer(dim)

    def forward(self, x, H, W):
        x = x.permute(0,2,3,1).contiguous()
        x = F.gelu(x)
        x = self.norm(x)
        x=x.permute(0,3,1,2)
        x=self.reduction(x)
        return x

class Patch_Expanding(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(dim)
        self.up=nn.ConvTranspose2d(dim,dim//2,2,2)
    def forward(self, x, H, W):
        x = x.permute(0,2,3,1).contiguous()
        x = self.norm(x)
        x = x.permute(0,3,1,2)
        x = self.up(x)
        return x

class BasicLayer(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size=7,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=True,
                 ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.dim=dim
        # build blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                i_block=i
                )
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1

        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1,
                                          self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            x = blk(x,attn_mask)

        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x,  H, W, x_down, Wh, Ww
        else:
            return x,  H, W, x, H, W

class BasicLayer_up(nn.Module):

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size=7,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 upsample=True
                ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.dim=dim

        # build blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                i_block=i)
            for i in range(depth)])

        self.Upsample = upsample(dim=2*dim, norm_layer=norm_layer)
    def forward(self, x,skip, H, W):
        x_up = self.Upsample(x, H, W)
        x = x_up + skip
        H, W = H * 2, W * 2
        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1

        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0

        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1,
                                          self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            x = blk(x,attn_mask)

        return x, H, W

class project(nn.Module):
    def __init__(self,in_dim,out_dim,stride,padding,activate,norm,last=False):
        super().__init__()
        self.out_dim=out_dim
        self.conv1=nn.Conv2d(in_dim,out_dim,kernel_size=3,stride=stride,padding=padding)
        self.conv2=nn.Conv2d(out_dim,out_dim,kernel_size=3,stride=1,padding=1)
        self.activate=activate()
        self.norm1=norm(out_dim)
        self.last=last
        if not last:
            self.norm2=norm(out_dim)

    def forward(self,x):
        x=self.conv1(x)
        x=self.activate(x)
        #norm1
        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm1(x)
        x = x.transpose(1, 2).view(-1, self.out_dim, Wh, Ww)
        x=self.conv2(x)
        if not self.last:
            x=self.activate(x)
            #norm2
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm2(x)
            x = x.transpose(1, 2).view(-1, self.out_dim, Wh, Ww)
        return x

class project_up(nn.Module):
    def __init__(self,in_dim,out_dim,activate,norm,last=False):
        super().__init__()
        self.out_dim=out_dim
        self.conv1=nn.ConvTranspose2d(in_dim,out_dim,kernel_size=2,stride=2)
        self.conv2=nn.Conv2d(out_dim,out_dim,kernel_size=3,stride=1,padding=1)
        self.activate=activate()
        self.norm1=norm(out_dim)
        self.last=last
        if not last:
            self.norm2=norm(out_dim)

    def forward(self,x):
        x=self.conv1(x)
        x=self.activate(x)
        #norm1
        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm1(x)
        x = x.transpose(1, 2).view(-1, self.out_dim, Wh, Ww)


        x=self.conv2(x)
        if not self.last:
            x=self.activate(x)
            #norm2
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm2(x)
            x = x.transpose(1, 2).view(-1, self.out_dim, Wh, Ww)
        return x



class PatchEmbed(nn.Module):

    def __init__(self, patch_size=4, in_chans=4, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_block=int(np.log2(patch_size[0]))
        self.project_block=[]
        self.dim=[int(embed_dim)//(2**i) for i in range(self.num_block)]
        self.dim.append(in_chans)
        self.dim=self.dim[::-1] # in_ch, embed_dim/2, embed_dim or in_ch, embed_dim/4, embed_dim/2, embed_dim

        for i in range(self.num_block)[:-1]:
            self.project_block.append(project(self.dim[i],self.dim[i+1],2,1,nn.GELU,nn.LayerNorm,False))
        self.project_block.append(project(self.dim[-2],self.dim[-1],2,1,nn.GELU,nn.LayerNorm,True))
        self.project_block=nn.ModuleList(self.project_block)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, self.patch_size[0] - W % self.patch_size[0]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        for blk in self.project_block:
            x = blk(x)

        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x



class encoder(nn.Module):
    def __init__(self,
                 pretrain_img_size=[224,224],
                 patch_size=[4,4],
                 in_chans=1  ,
                 embed_dim=96,
                 depths=[3, 3, 3, 3],
                 num_heads=[3, 6, 12, 24],
                 window_size=[7,7,14,7],
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 ):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices

        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    pretrain_img_size[0] // patch_size[0] // 2 ** i_layer, pretrain_img_size[1] // patch_size[1] // 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size[i_layer],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_path=dpr[sum(
                    depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging
                if (i_layer < self.num_layers - 1) else None,
                )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)


    def forward(self, x):
        """Forward function."""

        x = self.patch_embed(x)
        down=[]

        Wh, Ww = x.size(2), x.size(3)

        x = self.pos_drop(x)


        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out=x_out.permute(0,2,3,1)
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0,3, 1, 2).contiguous()

                down.append(out)
        return down


class decoder(nn.Module):
    def __init__(self,
                 pretrain_img_size,
                 embed_dim,
                 patch_size=[4,4],
                 depths=[3,3,3],
                 num_heads=[24,12,6],
                 window_size=[14,7,7],
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm
                 ):
        super().__init__()

        self.num_layers = len(depths)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers)[::-1]:

            layer = BasicLayer_up(
                dim=int(embed_dim * 2 ** (len(depths)-i_layer-1)),
                input_resolution=(
                    pretrain_img_size[0] // patch_size[0] // 2 ** (len(depths)-i_layer-1), pretrain_img_size[1] // patch_size[1] // 2 ** (len(depths)-i_layer-1)),

                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size[i_layer],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_path=dpr[sum(
                    depths[:(len(depths)-i_layer-1)]):sum(depths[:(len(depths)-i_layer)])],
                norm_layer=norm_layer,
                upsample=Patch_Expanding
                )
            self.layers.append(layer)
        self.num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
    def forward(self,x,skips):
        outs=[]
        H, W = x.size(2), x.size(3)
        x = self.pos_drop(x)

        for i in range(self.num_layers)[::-1]:
            layer = self.layers[i]
            x, H, W,  = layer(x,skips[i], H, W)
            outs.append(x)
        return outs





class final_patch_expanding(nn.Module):
    def __init__(self,dim,num_class,patch_size):
        super().__init__()
        self.num_block=int(np.log2(patch_size[0]))-2
        self.project_block=[]
        self.dim_list=[int(dim)//(2**i) for i in range(self.num_block+1)]
        # dim, dim/2, dim/4
        for i in range(self.num_block):
            self.project_block.append(project_up(self.dim_list[i],self.dim_list[i+1],nn.GELU,nn.LayerNorm,False))
        self.project_block=nn.ModuleList(self.project_block)
        self.up_final=nn.ConvTranspose2d(self.dim_list[-1],num_class,4,4)

    def forward(self,x):
        for blk in self.project_block:
            x = blk(x)
        x = self.up_final(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, input_resolution=None,num_heads=None,window_size=None,i_block=None,qkv_bias=None,qk_scale=None):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.blocks_tr = MSABlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i_block % 2 == 0) else window_size // 2,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=0,
                attn_drop=0,
                drop_path=drop_path)

    def forward(self, x,mask):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)

        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)

        x = x.permute(0,2,3,1).contiguous()
        x = self.blocks_tr(x,mask)
        x = x.permute(0,3,1,2).contiguous()

        return x



class unet2022(nn.Module):
    def __init__(self,
                 config = CONFIGS['ACDC_224'],
                 num_input_channels = 1,
                 embedding_dim = 96,
                 num_heads = [3, 6, 12, 24],
                 num_classes = 2,
                 deep_supervision = not True,
                 conv_op=nn.Conv2d):
        super(unet2022, self).__init__()

        # Don't uncomment conv_op
        self.num_input_channels = num_input_channels
        self.num_classes = num_classes
        self.conv_op = conv_op
        self.do_ds = deep_supervision
        self.embed_dim = embedding_dim
        self.depths=config.hyper_parameter.blocks_num
        self.num_heads=num_heads
        self.crop_size = config.hyper_parameter.crop_size
        self.patch_size=[config.hyper_parameter.convolution_stem_down,config.hyper_parameter.convolution_stem_down]
        self.window_size = config.hyper_parameter.window_size
        # if window size of the encoder is [7,7,14,7], then decoder's is [14,7,7]. In short, reverse the list and start from the index of 1
        self.model_down = encoder(
                                  pretrain_img_size=self.crop_size,
                                  window_size = self.window_size,
                                  embed_dim=self.embed_dim,
                                  patch_size=self.patch_size,
                                  depths=self.depths,
                                  num_heads=self.num_heads,
                                  in_chans=self.num_input_channels
                                 )

        self.decoder = decoder(
                               pretrain_img_size=self.crop_size,
                               window_size = self.window_size[::-1][1:],
                               embed_dim=self.embed_dim,
                               patch_size=self.patch_size,
                               depths=self.depths[::-1][1:],
                               num_heads=self.num_heads[::-1][1:]
                              )

        self.final=[]
        for i in range(len(self.depths)-1):
            self.final.append(final_patch_expanding(self.embed_dim*2**i,self.num_classes,patch_size=self.patch_size))
        self.final=nn.ModuleList(self.final)

    def forward(self, x):
        seg_outputs=[]
        skips = self.model_down(x)
        neck=skips[-1]
        out=self.decoder(neck,skips)

        for i in range(len(out)):
            seg_outputs.append(self.final[-(i+1)](out[i]))
        if self.do_ds:
            # for training
            return seg_outputs[::-1]
            #size [[224,224],[112,112],[56,56]]

        else:
            #for validation and testing
            return seg_outputs[-1]
            #size [[224,224]]

if __name__ == '__main__':
    # https: // blog.csdn.net / m0_73161433 / article / details / 134036788
    import torch
    from torchsummary import summary
    from thop import profile
    from thop import clever_format
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = unet2022().to(device)
    input_shape = (1, 1, 224, 224)  # (batch_size, channels, height, width, depth)
    input= torch.randn(input_shape).to(device)
    print("input_shape: ", input.shape)
    output = model(input)
    print("output_shape: ", output.shape)
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")  # 格式化显示FLOPs和参数量
    print(f"FLOPs: {flops}, Params: {params}")
    # summary(model, input_size=(1, 512, 512), device=device.type) # summary 返回值有时存在很多问题
    # print(model)
    # print(model.state_dict().keys())
    # print(model.state_dict().keys())




