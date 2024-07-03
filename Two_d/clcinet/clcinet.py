import torch
import torch.nn as nn
from torch import autograd
import torch.nn.functional as F
import sys
from .ConvLSTM import ConvLSTM

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
 
    def forward(self, input):
        return self.conv(input)

class conv_1_init(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size = 3, stride = 1, padding = 1):
        super(conv_1_init, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size = kernel_size, stride = stride, padding = padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class dilate_conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size = 3, dilation_rate = 1, padding = 1):
        super(dilate_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size = kernel_size, dilation = dilation_rate, padding = padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class concat_pool(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size = 3, stride = 2, padding = 1):
        super(concat_pool, self).__init__()
        self.conv_pool = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size = kernel_size, stride = stride, padding = padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace = True)
        )

    def forward(self, conv, pool):
        conv = self.conv_pool(conv)
        cat = torch.cat([conv, pool], dim = 1)
        return cat
class conv_lstm(nn.Module):
    def __init__(self, channel = 256):
        super(conv_lstm, self).__init__()
        self.lstm = ConvLSTM(channel, hidden_dim = channel,  kernel_size=(3, 3), num_layers=1)
        self.conv = nn.Conv2d(channel*2, channel, kernel_size=1)
    def forward(self, input):
        
        input = self.conv(input)
        input = input.unsqueeze(dim = 0)
        output = self.lstm(input)
        return output



class CLCInet(nn.Module):
    def __init__(self, in_ch=1, out_ch=2, is_train = False):
        super(CLCInet, self).__init__()
        self.is_train = is_train
        self.conv1 = DoubleConv(in_ch, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.concat_pool1 = concat_pool(32, 32, kernel_size=3, stride = 2)
        # self.concat_pool1 = nn.Sequential(
        #     nn.Conv2d(32, 32, 3, stride=2, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True)
        # )
        self.conv_fusion1 = conv_1_init(64, 256, kernel_size = 1, padding=0)
        self.conv2 = DoubleConv(256, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.concat_pool12 = concat_pool(32, 64, stride=4)
        self.concat_pool22 = concat_pool(64, 64, stride=2)
        self.conv_fusion2 = conv_1_init(64*3, 512, kernel_size=1, padding=0)

        self.conv3 = DoubleConv(512, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.concat_pool13 = concat_pool(32, 128, stride=8)
        self.concat_pool23 = concat_pool(64, 128, stride=4)
        self.concat_pool33 = concat_pool(128, 128)
        self.conv_fusion3 = conv_1_init(128*4, 1024, kernel_size=1, padding=0)

        self.conv4 = DoubleConv(1024, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.concat_pool14 = concat_pool(32, 256, stride=16)
        self.concat_pool24 = concat_pool(64, 256, stride=8)
        self.concat_pool34 = concat_pool(128, 256, stride=4)
        self.concat_pool44 = concat_pool(256, 256, stride=2)
        self.conv_fusion4 = conv_1_init(256*5, 2048, kernel_size=1, padding=0)

        self.conv5 = DoubleConv(2048, 512)


        #### ASPP 9*256
        self.b0 = conv_1_init(512, 256, 1, padding=0)
        self.b1 = dilate_conv(512, 256, dilation_rate=2, padding=2)
        self.b2 = dilate_conv(512, 256, dilation_rate=4, padding=4)
        self.b3 = dilate_conv(512, 256, dilation_rate=6, padding=6)
        self.b4_pool = nn.AvgPool2d(2) #全局池化
        self.b4_convinit = conv_1_init(512, 256, 1, padding=0)

        self.clf1 = conv_1_init(32, 256, stride=16)
        self.clf2 = conv_1_init(64, 256, stride=8)
        self.clf3 = conv_1_init(128, 256, stride=4)
        self.clf4 = conv_1_init(256, 256, stride=2)

        self.last_conv = conv_1_init(256*9, 1024, kernel_size=1, padding=0)
        ####

        self.up1 = nn.UpsamplingNearest2d(scale_factor=2) ## equal to 'keras.layer.UpSampling2D'
        self.up1_conv = conv_1_init(1024, 256, kernel_size=3)
        self.c4_conv = conv_1_init(256, 256, kernel_size=1, padding=0)
        self.context_inference1 = conv_lstm(channel=256)
        
        self.conv6 = DoubleConv(256, 256)
        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up2_conv = conv_1_init(256, 128, kernel_size=3)
        self.c3_conv = conv_1_init(128, 128, kernel_size=1, padding=0)
        self.context_inference2 = conv_lstm(channel=128)

        self.conv7 = DoubleConv(128, 128)
        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up3_conv = conv_1_init(128, 64, kernel_size=3)
        self.c2_conv = conv_1_init(64, 64, kernel_size=1, padding=0)
        self.context_inference3 = conv_lstm(channel=64)

        self.conv8 = DoubleConv(64, 64)
        self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up4_conv = conv_1_init(64, 32, kernel_size=3)
        self.c1_conv = conv_1_init(32, 32, kernel_size=1, padding=0)
        self.context_inference4 = conv_lstm(channel=32)

        self.conv9 = DoubleConv(32, 32)
        # self.conv10 = nn.Sequential(
        #     nn.Conv2d(32, 1, kernel_size=1),
        #     nn.Sigmoid()
        # )
        self.conv10 = nn.Conv2d(32, out_ch, kernel_size=1)

    def CLF_APSS(self, c5, c1, c2, c3, c4):
        b0 = self.b0(c5)
        # print('b0:', b0.shape) # 8, 256, 16, 16
        b1 = self.b1(c5)
        # print('b1:', b1.shape) # 8, 256, 16, 16
        b2 = self.b2(c5)
        # print('b2:', b2.shape) # 8, 256, 16, 16
        b3 = self.b3(c5)
        # print('b3:', b3.shape) # 8, 256, 16, 16
        b4 = self.b4_pool(c5)
        
        # x, size, mode='bilinear', align_corners=True
        b4 = self.b4_convinit(b4)
        b4 = F.interpolate(b4, (b0.size()[2], b0.size()[3]), mode = 'bilinear', align_corners=True)
        # print("b4:", b4.shape) # 8, 256, 16, 16
        clf1 = self.clf1(c1)
        clf2 = self.clf2(c2)
        clf3 = self.clf3(c3)
        clf4 = self.clf4(c4)
        # print(clf1.shape, clf2.shape, clf3.shape, clf4.shape)
        res = torch.cat([clf1, clf2, clf3, clf4, b0, b1, b2, b3, b4],dim = 1)
        res = self.last_conv(res)
        if self.is_train:
            res = nn.Dropout2d(0.5)(res)
        return res
    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        concat_pool11 = self.concat_pool1(c1, p1) # [8, 64, 128, 128]
        fusion1 = self.conv_fusion1(concat_pool11)
        
        c2 = self.conv2(fusion1)
        p2 = self.pool2(c2)
        # print(p2.shape, c1.shape)
        concat_pool12 = self.concat_pool12(c1, p2) 
        concat_pool22 = self.concat_pool22(c2, concat_pool12)
        fusion2 = self.conv_fusion2(concat_pool22)
        
        c3 = self.conv3(fusion2)
        p3 = self.pool3(c3)
        concat_pool13 = self.concat_pool13(c1, p3)
        concat_pool23 = self.concat_pool23(c2, concat_pool13)
        concat_pool33 = self.concat_pool33(c3, concat_pool23)
        fusion3 = self.conv_fusion3(concat_pool33)

        c4 = self.conv4(fusion3)
        p4 = self.pool4(c4)
        concat_pool14 = self.concat_pool14(c1, p4)
        concat_pool24 = self.concat_pool24(c2, concat_pool14)
        concat_pool34 = self.concat_pool34(c3, concat_pool24)
        concat_pool44 = self.concat_pool44(c4, concat_pool34)
        fusion4 = self.conv_fusion4(concat_pool44)
        # print(fusion4.shape) # 8, 2048, 16, 16

        c5 = self.conv5(fusion4)
        if self.is_train:    #  训练时加dropout 测试时不用
            c5 = nn.Dropout2d(0.5)(c5)
        # print(c5.shape) # [8, 512, 16, 16]
        
        clf_apss = self.CLF_APSS(c5, c1, c2, c3, c4)
        # print("clf_apss", clf_apss.shape) 8, 1024, 16, 16
        up_c1 = self.up1(clf_apss)
        # print("up_c1:",up_c1.shape)
        up_c1 = self.up1_conv(up_c1)
        skip_conv4 = self.c4_conv(c4)
        lstmconcat1 = torch.cat([up_c1, skip_conv4], dim=1)
        # print("lstmconcat1.shape", lstmconcat1.shape)
        context_inference1, _= self.context_inference1(lstmconcat1)
        # print(context_inference1[0].shape)
        context_inference1 = context_inference1[0].squeeze(1)
        # print("context_inference1.shape", context_inference1.shape) # 8, 256, 32, 32
        
        c6 = self.conv6(context_inference1)
        up_c2 = self.up2(c6)
        up_c2 = self.up2_conv(up_c2)
        skip_conv3 = self.c3_conv(c3)
        lstmconcat2 = torch.cat([up_c2, skip_conv3], dim = 1)
        # print("lstmconcat2.shape", lstmconcat2.shape) # 8, 256, 64, 64
        context_inference2, _ = self.context_inference2(lstmconcat2)
        context_inference2 = context_inference2[0].squeeze(1)

        c7 = self.conv7(context_inference2)
        # print("c7.shape", c7.shape)
        up_c3 = self.up3(c7)
        up_c3 = self.up3_conv(up_c3)
        skip_conv2 = self.c2_conv(c2)
        lstmconcat3 = torch.cat([up_c3, skip_conv2], dim = 1)
        context_inference3, _ = self.context_inference3(lstmconcat3)
        context_inference3 = context_inference3[0].squeeze(1)
        # print('context_inference3.shape:', context_inference3.shape) 8, 64, 128, 128
        # assert 1>3
        c8 = self.conv8(context_inference3)
        up_c4 = self.up4(c8)
        up_c4 = self.up4_conv(up_c4)
        skip_conv1 = self.c1_conv(c1)
        lstmconcat4 = torch.cat([up_c4, skip_conv1], dim = 1)
        context_inference4, _ = self.context_inference4(lstmconcat4)
        context_inference4 = context_inference4[0].squeeze(1)
        # print('context_inference4.shape', context_inference4.shape)
        c9 = self.conv9(context_inference4)
        out = self.conv10(c9)

        return out

