
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummaryX import summary
from Models1.sync_batchnorm import SynchronizedBatchNorm2d
from torchvision.models import resnet34, resnet50, resnet101, resnet152, resnet18
from Models1.batchnorm import SynchronizedBatchNorm2d
from einops import rearrange, reduce, repeat, parse_shape
from Models1.utils import conv_bn_relu



class PSNet(nn.Module):
    def __init__(self, backbone,  sync_bn=True, pretrained=True, ResNet34M= False, criterion=nn.CrossEntropyLoss(ignore_index=255), classes = 6):
        super(PSNet, self).__init__()
        self.ResNet34M = ResNet34M
        self.backbone = backbone
        self.criterion = criterion

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        if backbone.lower() == "resnet18":
            encoder = resnet18(pretrained=pretrained)
        elif backbone.lower() == "resnet34":
            encoder = resnet34(pretrained=pretrained)
        elif backbone.lower() == "resnet50":
            encoder = resnet50(pretrained=pretrained)
        elif backbone.lower() == "resnet101":
            encoder = resnet101(pretrained=pretrained)
        elif backbone.lower() == "resnet152":
            encoder = resnet152(pretrained=pretrained)
        else:
            raise NotImplementedError("{} Backbone not implemented".format(backbone))

        self.out_channels = [32,64,128,256,512,1024,2048]
        self.conv1_x = encoder.conv1
        self.bn1 = encoder.bn1
        self.relu = encoder.relu
        self.maxpool = encoder.maxpool
        self.conv2_x = encoder.layer1  # 1/4
        self.conv3_x = encoder.layer2  # 1/8
        self.conv4_x = encoder.layer3  # 1/16
        self.conv5_x = encoder.layer4  # 1/32

        self.decoder = MDPM(self.out_channels[4], self.out_channels[4], BatchNorm) #

        self.down2 = conv_block(self.out_channels[-4], self.out_channels[1], 3, 1, 1, 1, 1, bn_act=True)
        self.down3 = conv_block(self.out_channels[-3], self.out_channels[2], 3, 1, 1, 1, 1, bn_act=True)
        self.down4 = conv_block(self.out_channels[-2], self.out_channels[3], 3, 1, 1, 1, 1, bn_act=True)
        self.down5 = conv_block(self.out_channels[-1], self.out_channels[4], 3, 1, 1, 1, 1, bn_act=True)

        self.apf1 = FWM(self.out_channels[4], self.out_channels[4], self.out_channels[3], classes=classes)
        self.apf2 = FWM(self.out_channels[3], self.out_channels[3], self.out_channels[2], classes=classes)
        self.apf3 = FWM(self.out_channels[2], self.out_channels[2], self.out_channels[1], classes=classes)
        self.apf4 = FWM(self.out_channels[1], self.out_channels[1], self.out_channels[0], classes=classes)


        self.classifier = SegHead(self.out_channels[0], classes)

        self.D = LCGB(3, self.out_channels[1])
        self.saff = SWAM()

    def forward(self, x, y=None, z= None):
        B, C, H, W = x.size()

        DH = self.D(x) #torch.Size([6, 64, 160, 160])
        # print(DH.size())

        x = self.conv1_x(x)
        x = self.bn1(x)
        x1 = self.relu(x)
        x = self.maxpool(x1)
        if self.ResNet34M:
            x2 = self.conv2_x(x1)
        else:
            x2 = self.conv2_x(x)
        x3 = self.conv3_x(x2)
        x4 = self.conv4_x(x3)
        x5 = self.conv5_x(x4)

        if self.backbone in ['resnet50', 'resnet101', 'resnet152']:
            x2 = self.down2(x2)
            x3 = self.down3(x3)
            x4 = self.down4(x4)
            x5 = self.down5(x5)
            
        cfgb1 = self.decoder(x5) #torch.Size([6, 512, 16, 16])
        D112 = torch.add(DH, x2) #torch.Size([6, 64, 160, 160])

        APF1 = self.apf1(cfgb1, x5)
        APF2 = self.apf2(APF1, x4)
        APF3 = self.apf3(APF2, x3)
        APF4 = self.apf4(APF3, D112) #x2
        
        APF = self.saff(APF4)

        classifier1 = self.classifier(APF41)
        predict = F.interpolate(classifier1, size=(H, W), mode="bilinear", align_corners=True)
        
        if self.training:
            main_loss = self.criterion(predict, y)
            # axu_loss = 1/4 * self.criterion(s1, y) + 1/8 * self.criterion(s2, y) + 1/16 * self.criterion(s3, y) + 1/32 * self.criterion(s4, y)
            # main_loss = main_loss + axu_loss
            return predict.max(1)[1], main_loss, main_loss
        else:
            return predict

        return predict

