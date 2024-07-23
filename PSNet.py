
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
        super(SSFPN, self).__init__()
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

class SWAM(nn.Module):
    def __init__(self, a=0.5, b=2, sigma=0.0001):
        super(SWAM, self).__init__()
        self.a = a
        self.b = b
        self.sigma = sigma

    def forward(self, x):
        """
        :param x: (n, c, h, w)
        :return:
        """
        n, K, h, w = x.shape
        S = x.sum(dim=1)  # n,h,w
        z = torch.sum(S ** self.a, dim=[1, 2])
        z = (z ** (1 / self.a)).view(n, 1, 1)
        S = (S / z) ** (1 / self.b)
        S = S.unsqueeze(1)
        AW = (x * S).sum(dim=[2, 3])
        # print(new_x.size())
        AW = AW.unsqueeze(2).unsqueeze(3)
        AW = AW.expand_as(x)
        AW1 = AW * x
        new = x + AW1
        return new



class LCGB(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        inner_ch = out_ch // 2
        self.down = NeighborDecouple(4) #4

        self.conv = nn.Sequential(
            conv_bn_relu(4 * in_ch, out_ch, 3, groups=4),
            conv_bn_relu(out_ch, out_ch, 1),
            conv_bn_relu(out_ch, out_ch, 3, groups=4),
            conv_bn_relu(out_ch, out_ch, 1),
            conv_bn_relu(out_ch, out_ch, 3, groups=4),
            conv_bn_relu(out_ch, out_ch, 1),
            conv_bn_relu(out_ch, out_ch, 3, groups=4),
            conv_bn_relu(out_ch, out_ch, 1),
        )

        self.conv1 = nn.Sequential(
            conv_bn_relu(16 * in_ch, out_ch, 3, groups=16),
            conv_bn_relu(out_ch, out_ch, 1),
            conv_bn_relu(out_ch, out_ch, 3, groups=16),
            conv_bn_relu(out_ch, out_ch, 1),
            conv_bn_relu(out_ch, out_ch, 3, groups=16),
            conv_bn_relu(out_ch, out_ch, 1),
            conv_bn_relu(out_ch, out_ch, 3, groups=16),
            conv_bn_relu(out_ch, out_ch, 1),
        )

        self.avgPool = nn.AvgPool2d(9, stride=8, padding=4, count_include_pad=False)
        self.to_qkv = conv_bn_relu(out_ch, 3 * inner_ch, 1)

        self.expand = conv_bn_relu(out_ch, out_ch, 1)

    def forward(self, x):
        # print(x.size())
        x = self.down(x)
        # print(x.size())
        x = self.conv1(x)
        # print(x.size())

        _, _, h, w = x.shape
        pooled = self.avgPool(x)

        qkv = self.to_qkv(pooled).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b c h w -> b c (h w)'), qkv)
        out = torch.einsum('bkl,bkt->blt', [k, q])
        out = F.softmax(out * (q.shape[1] ** (-0.5)), dim=1)
        out = torch.einsum('blt,btv->blv', [v, out])
        out = torch.cat([out, q], 1)
        out = rearrange(out, 'b c (h w) -> b c h w',
                        **parse_shape(pooled, 'b c h w'))
        out = self.expand(out)

        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
        x = x + out

        return x


class NeighborDecouple(nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        x = rearrange(x, 'b c (h h2) (w w2) -> b (h2 w2 c) h w', h2=self.bs, w2=self.bs)
        return x
# 全局
class MDPM(nn.Module):
    def __init__(self, in_channels, n_filters, BatchNorm, inp=False):
        super(MDPM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.bn1 = BatchNorm(in_channels // 4)
        self.relu1 = nn.ReLU()
        self.inp = inp

        self.deconv1 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4)
        )
        self.deconv2 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0)
        )
        self.deconv3 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0)
        )
        self.deconv4 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4)
        )

        self.bn2 = BatchNorm(in_channels // 4 + in_channels // 4)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            in_channels // 4 + in_channels // 4, n_filters, 1)
        self.bn3 = BatchNorm(n_filters)
        self.relu3 = nn.ReLU()

        self._init_weight()

    def forward(self, x, inp = False):
        # print(x.size())  #torch.Size([6, 512, 20, 20])
        # print("wwwwwwwww")
        x = self.conv1(x) #torch.Size([6, 128, 20, 20])

        x = self.bn1(x)  #torch.Size([6, 128, 20, 20])

        x = self.relu1(x)#torch.Size([6, 128, 20, 20])


        x1 = self.deconv1(x)  #torch.Size([6, 64, 20, 20])

        x2 = self.deconv2(x)  #torch.Size([6, 64, 20, 20])

        # x31 = self.h_transform(x) #torch.Size([6, 128, 20, 39])
        # print(x31.size())
        # x32 = self.deconv3(x31) #torch.Size([6, 64, 20, 39])
        # print(x32.size())
        # x33 = self.inv_h_transform(x32) #torch.Size([6, 64, 20, 20])
        # print(x33.size())

        x3 = self.inv_h_transform(self.deconv3(self.h_transform(x)))
        x4 = self.inv_v_transform(self.deconv4(self.v_transform(x)))
        x = torch.cat((x1, x2, x3, x4), 1)
        if self.inp:
            x = F.interpolate(x, scale_factor=2)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def h_transform(self, x):
        shape = x.size() #torch.Size([6, 128, 20, 20])

        x = torch.nn.functional.pad(x, (0, shape[-1]))   #torch.Size([6, 128, 20, 40])

        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]] # torch.Size([6, 128, 780])

        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1) #torch.Size([6, 128, 20, 39])

        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)
        
class FWM(nn.Module):
    def __init__(self, channels_high, channels_low, channel_out, classes=11):
        super(FWM, self).__init__()

        self.lateral_low = conv_block(channels_low, channels_high, 1, 1, bn_act=True, padding=0)

        self.conv_low = conv_block(channels_high, channel_out, 3, 1, bn_act=True, padding=1)
        self.sa = SpatialAttention(channel_out, channel_out)

        self.conv_high = conv_block(channels_high, channel_out, 3, 1, bn_act=True, padding=1)
        self.ca = ChannelWise(channel_out)

        self.FRB = nn.Sequential(
            conv_block(2 * channels_high, channel_out, 1, 1, bn_act=True, padding=0),
            conv_block(channel_out, channel_out, 3, 1, bn_act=True, group=1, padding=1))

        self.classifier = nn.Sequential(
            conv_block(channel_out, channel_out, 3, 1, padding=1, group=1, bn_act=True),
            nn.Dropout(p=0.15),
            conv_block(channel_out, classes, 1, 1, padding=0, bn_act=False))
        self.apf = conv_block(channel_out, channel_out, 3, 1, padding=1, group=1, bn_act=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.active = nn.Sigmoid()

    def forward(self, x_high, x_low):
        _, _, h, w = x_low.size()

        lat_low = self.lateral_low(x_low)

        high_up1 = F.interpolate(x_high, size=lat_low.size()[2:], mode='bilinear', align_corners=False)

        concate = torch.cat([lat_low, high_up1], 1)
        concate = self.FRB(concate)

        conv_high = self.conv_high(high_up1)
        conv_low = self.conv_low(lat_low)

        sa = self.sa(concate)
        ca = self.ca(concate)

        mul1 = torch.mul(sa, conv_high)
        mul2 = torch.mul(ca, conv_low)

        # mul1 = mul1 + conv_high
        # mul2 = mul2 + conv_low
        att_out = mul1 + mul2

        cat_f = torch.cat((conv_high, conv_low, att_out), dim=1)
        b, mc, _, _ = cat_f.size()
        act_f = self.active(torch.abs(cat_f))
        avg_f = self.avg_pool(act_f)
        weight = torch.split(avg_f, mc // 3, dim=1)
        out = weight[0] * conv_high + weight[1] * conv_low + weight[2] * (att_out)
        att_out =out

        APF = self.apf(att_out)
        return APF
