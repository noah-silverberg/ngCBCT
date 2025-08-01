import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
import ast
from tqdm import tqdm
import copy
import torch.distributions as distributions


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('InstanceNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class SingleConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class DownConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(DownConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_in, ch_in, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=3,
                      stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, up_conv):
        super(UpConvBlock, self).__init__()

        if up_conv:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(ch_in, ch_out, kernel_size=3,
                                   stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(ch_out),
                nn.ReLU(inplace=True)
            )

        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(ch_in, ch_out, kernel_size=3,
                          stride=1, padding=1, bias=True),
                nn.InstanceNorm2d(ch_out),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.up(x)
        return x


class RecurrentBlock(nn.Module):
    def __init__(self, ch_out, t=2):
        super(RecurrentBlock, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x+x1)
        return x1


class RRCNNBlock(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNNBlock, self).__init__()
        self.RCNN = nn.Sequential(
            RecurrentBlock(ch_out, t=t),
            RecurrentBlock(ch_out, t=t)
        )
        self.conv_1x1 = nn.Conv2d(
            ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1


class ResidualBlock(nn.Module):
    # original ResNet Block
    def __init__(self, ch_in):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            # Pads the input tensor using the reflection of the input boundary
            nn.Conv2d(ch_in, ch_in, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_in),
            # nn.InstanceNorm2d(ch_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_in, ch_in, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_in)
            # nn.InstanceNorm2d(ch_in),
        )
        self.relu = nn.Sequential(
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        y = x + self.block(x)
        y1 = self.relu(y)
        return y1


class ResidualBlock_mod(nn.Module):
    # ReflectionPad and InstanceNorm
    def __init__(self, ch_in):
        super(ResidualBlock_mod, self).__init__()

        self.block = nn.Sequential(
            # Pads the input tensor using the reflection of the input boundary
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch_in, ch_in, 3),
            nn.InstanceNorm2d(ch_in),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch_in, ch_in, 3),
            nn.InstanceNorm2d(ch_in)
        )
        self.relu = nn.Sequential(
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        y = x + self.block(x)
        y1 = self.relu(y)
        return y1


class ResidualBlock_mod_2(nn.Module):
    # without ReLU after concat.

    def __init__(self, ch_in):
        super(ResidualBlock_mod_2, self).__init__()

        self.block = nn.Sequential(
            # Pads the input tensor using the reflection of the input boundary
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch_in, ch_in, 3),
            nn.InstanceNorm2d(ch_in),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch_in, ch_in, 3),
            nn.InstanceNorm2d(ch_in)
        )

    def forward(self, x):
        y = x + self.block(x)
        return y


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class UNet(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(UNet, self).__init__()

        up_conv = True

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = ConvBlock(ch_in=img_ch, ch_out=64)
        self.conv2 = ConvBlock(ch_in=64, ch_out=128)
        self.conv3 = ConvBlock(ch_in=128, ch_out=256)
        self.conv4 = ConvBlock(ch_in=256, ch_out=512)
        self.conv5 = ConvBlock(ch_in=512, ch_out=1024)

        self.up5 = UpConvBlock(ch_in=1024, ch_out=512, up_conv=up_conv)
        self.up_conv5 = ConvBlock(ch_in=1024, ch_out=512)

        self.up4 = UpConvBlock(ch_in=512, ch_out=256, up_conv=up_conv)
        self.up_conv4 = ConvBlock(ch_in=512, ch_out=256)

        self.up3 = UpConvBlock(ch_in=256, ch_out=128, up_conv=up_conv)
        self.up_conv3 = ConvBlock(ch_in=256, ch_out=128)

        self.up2 = UpConvBlock(ch_in=128, ch_out=64, up_conv=up_conv)
        self.up_conv2 = ConvBlock(ch_in=128, ch_out=64)

        self.conv_1x1 = nn.Conv2d(
            64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.conv1(x)

        x2 = self.maxpool(x1)
        x2 = self.conv2(x2)

        x3 = self.maxpool(x2)
        x3 = self.conv3(x3)

        x4 = self.maxpool(x3)
        x4 = self.conv4(x4)

        x5 = self.maxpool(x4)
        x5 = self.conv5(x5)

        # decoding + concat path
        d5 = self.up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.up_conv5(d5)

        d4 = self.up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_conv2(d2)

        d1 = self.conv_1x1(d2)

        # additional skip connection between input and output
        result = x + d1

        return result


class R2UNet(nn.Module):
    def __init__(self, img_ch=1, output_ch=1, t=2):
        super(R2UNet, self).__init__()

        up_conv = True

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNNBlock(ch_in=img_ch, ch_out=64, t=t)

        self.RRCNN2 = RRCNNBlock(ch_in=64, ch_out=128, t=t)

        self.RRCNN3 = RRCNNBlock(ch_in=128, ch_out=256, t=t)

        self.RRCNN4 = RRCNNBlock(ch_in=256, ch_out=512, t=t)

        self.RRCNN5 = RRCNNBlock(ch_in=512, ch_out=1024, t=t)

        self.up5 = UpConvBlock(ch_in=1024, ch_out=512, up_conv=up_conv)
        self.up_RRCNN5 = RRCNNBlock(ch_in=1024, ch_out=512, t=t)

        self.up4 = UpConvBlock(ch_in=512, ch_out=256, up_conv=up_conv)
        self.up_RRCNN4 = RRCNNBlock(ch_in=512, ch_out=256, t=t)

        self.up3 = UpConvBlock(ch_in=256, ch_out=128, up_conv=up_conv)
        self.up_RRCNN3 = RRCNNBlock(ch_in=256, ch_out=128, t=t)

        self.up2 = UpConvBlock(ch_in=128, ch_out=64, up_conv=up_conv)
        self.up_RRCNN2 = RRCNNBlock(ch_in=128, ch_out=64, t=t)

        self.conv_1x1 = nn.Conv2d(
            64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.up_RRCNN5(d5)

        d4 = self.up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.up_RRCNN4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_RRCNN3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_RRCNN2(d2)

        d1 = self.conv_1x1(d2)

        # additional skip connection between input and output
        result = x + d1

        return result


class AttUNet(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(AttUNet, self).__init__()

        up_conv = True

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = ConvBlock(ch_in=img_ch, ch_out=64)
        self.conv1_extra = SingleConv(ch_in=64, ch_out=64)
        self.conv2 = ConvBlock(ch_in=64, ch_out=128)
        self.conv3 = ConvBlock(ch_in=128, ch_out=256)
        self.conv4 = ConvBlock(ch_in=256, ch_out=512)
        self.conv5 = ConvBlock(ch_in=512, ch_out=1024)

        self.up5 = UpConvBlock(ch_in=1024, ch_out=512, up_conv=up_conv)
        self.att5 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.up_conv5 = ConvBlock(ch_in=1024, ch_out=512)

        self.up4 = UpConvBlock(ch_in=512, ch_out=256, up_conv=up_conv)
        self.att4 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.up_conv4 = ConvBlock(ch_in=512, ch_out=256)

        self.up3 = UpConvBlock(ch_in=256, ch_out=128, up_conv=up_conv)
        self.att3 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.up_conv3 = ConvBlock(ch_in=256, ch_out=128)

        self.up2 = UpConvBlock(ch_in=128, ch_out=64, up_conv=up_conv)
        self.att2 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.up_conv2 = ConvBlock(ch_in=128, ch_out=64)

        self.conv_1x1 = nn.Conv2d(
            64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, input):
        # encoding path
        e1 = self.conv1(input)
        e1 = self.conv1_extra(e1)

        e2 = self.maxpool(e1)
        e2 = self.conv2(e2)

        e3 = self.maxpool(e2)
        e3 = self.conv3(e3)

        e4 = self.maxpool(e3)
        e4 = self.conv4(e4)

        e5 = self.maxpool(e4)
        e5 = self.conv5(e5)

        # decoding + concat path
        d5 = self.up5(e5)
        e4 = self.att5(g=d5, x=e4)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.up_conv5(d5)

        d4 = self.up4(d5)
        # e3 = self.att4(g=d4,x=e3)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        # e2 = self.att3(g=d3,x=e2)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        # e1 = self.att2(g=d2,x=e1)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.up_conv2(d2)

        d1 = self.conv_1x1(d2)

        # additional skip connection between input and output
        result = input + d1

        return result


class R2AttUNet(nn.Module):
    def __init__(self, img_ch=1, output_ch=1, t=2):
        super(R2AttUNet, self).__init__()

        up_conv = True

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNNBlock(ch_in=img_ch, ch_out=64, t=t)

        self.RRCNN2 = RRCNNBlock(ch_in=64, ch_out=128, t=t)

        self.RRCNN3 = RRCNNBlock(ch_in=128, ch_out=256, t=t)

        self.RRCNN4 = RRCNNBlock(ch_in=256, ch_out=512, t=t)

        self.RRCNN5 = RRCNNBlock(ch_in=512, ch_out=1024, t=t)

        self.up5 = UpConvBlock(ch_in=1024, ch_out=512, up_conv=up_conv)
        self.att5 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.up_RRCNN5 = RRCNNBlock(ch_in=1024, ch_out=512, t=t)

        self.up4 = UpConvBlock(ch_in=512, ch_out=256, up_conv=up_conv)
        self.att4 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.up_RRCNN4 = RRCNNBlock(ch_in=512, ch_out=256, t=t)

        self.up3 = UpConvBlock(ch_in=256, ch_out=128, up_conv=up_conv)
        self.att3 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.up_RRCNN3 = RRCNNBlock(ch_in=256, ch_out=128, t=t)

        self.up2 = UpConvBlock(ch_in=128, ch_out=64, up_conv=up_conv)
        self.att2 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.up_RRCNN2 = RRCNNBlock(ch_in=128, ch_out=64, t=t)

        self.conv_1x1 = nn.Conv2d(
            64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.up5(x5)
        x4 = self.att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.up_RRCNN5(d5)

        d4 = self.up4(d5)
        x3 = self.att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.up_RRCNN4(d4)

        d3 = self.up3(d4)
        x2 = self.att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_RRCNN3(d3)

        d2 = self.up2(d3)
        x1 = self.att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_RRCNN2(d2)

        d1 = self.conv_1x1(d2)

        # additional skip connection between input and output
        result = x + d1

        return result


class FBPCONVNet(nn.Module):
    """FBPCONVNet"""

    def __init__(self, img_ch=1, output_ch=1):
        super(FBPCONVNet, self).__init__()

        up_conv = True

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = ConvBlock(ch_in=img_ch, ch_out=64)
        self.conv1_extra = SingleConv(ch_in=64, ch_out=64)
        self.conv2 = ConvBlock(ch_in=64, ch_out=128)
        self.conv3 = ConvBlock(ch_in=128, ch_out=256)
        self.conv4 = ConvBlock(ch_in=256, ch_out=512)
        self.conv5 = ConvBlock(ch_in=512, ch_out=1024)

        self.up5 = UpConvBlock(ch_in=1024, ch_out=512, up_conv=up_conv)
        self.up_conv5 = ConvBlock(ch_in=1024, ch_out=512)
        self.up4 = UpConvBlock(ch_in=512, ch_out=256, up_conv=up_conv)
        self.up_conv4 = ConvBlock(ch_in=512, ch_out=256)
        self.up3 = UpConvBlock(ch_in=256, ch_out=128, up_conv=up_conv)
        self.up_conv3 = ConvBlock(ch_in=256, ch_out=128)
        self.up2 = UpConvBlock(ch_in=128, ch_out=64, up_conv=up_conv)
        self.up_conv2 = ConvBlock(ch_in=128, ch_out=64)

        self.conv_1x1 = nn.Conv2d(
            64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, input):
        # encoding path
        e1 = self.conv1(input)
        e1 = self.conv1_extra(e1)

        e2 = self.maxpool(e1)
        e2 = self.conv2(e2)

        e3 = self.maxpool(e2)
        e3 = self.conv3(e3)

        e4 = self.maxpool(e3)
        e4 = self.conv4(e4)

        e5 = self.maxpool(e4)
        e5 = self.conv5(e5)

        # decoding + concat path
        d5 = self.up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.up_conv5(d5)

        d4 = self.up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.up_conv2(d2)

        d1 = self.conv_1x1(d2)

        # additional skip connection between input and output
        result = input + d1

        return result


class FBPCONVNet_dropout(nn.Module):
    """FBPCONVNet"""

    def __init__(self, img_ch=1, output_ch=1):
        super(FBPCONVNet_dropout, self).__init__()

        up_conv = True

        self.dropout = nn.Dropout(0.15)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = ConvBlock(ch_in=img_ch, ch_out=64)
        self.conv1_extra = SingleConv(ch_in=64, ch_out=64)
        self.conv2 = ConvBlock(ch_in=64, ch_out=128)
        self.conv3 = ConvBlock(ch_in=128, ch_out=256)
        self.conv4 = ConvBlock(ch_in=256, ch_out=512)
        self.conv5 = ConvBlock(ch_in=512, ch_out=1024)

        self.up5 = UpConvBlock(ch_in=1024, ch_out=512, up_conv=up_conv)
        self.up_conv5 = ConvBlock(ch_in=1024, ch_out=512)
        self.up4 = UpConvBlock(ch_in=512, ch_out=256, up_conv=up_conv)
        self.up_conv4 = ConvBlock(ch_in=512, ch_out=256)
        self.up3 = UpConvBlock(ch_in=256, ch_out=128, up_conv=up_conv)
        self.up_conv3 = ConvBlock(ch_in=256, ch_out=128)
        self.up2 = UpConvBlock(ch_in=128, ch_out=64, up_conv=up_conv)
        self.up_conv2 = ConvBlock(ch_in=128, ch_out=64)

        self.conv_1x1 = nn.Conv2d(
            64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, input):
        # encoding path
        e1 = self.conv1(input)
        e1 = self.conv1_extra(e1)

        e2 = self.maxpool(e1)
        e2 = self.conv2(e2)

        e3 = self.maxpool(e2)
        e3 = self.conv3(e3)

        e4 = self.maxpool(e3)
        e4 = self.conv4(e4)

        e5 = self.maxpool(e4)
        e5 = self.conv5(e5)

        # decoding + concat path
        d5 = self.up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.up_conv5(d5)

        d4 = self.up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.up_conv2(d2)

        d1 = self.conv_1x1(d2)

        # additional skip connection between input and output
        result = input + d1

        return result


class IResNet(nn.Module):

    def __init__(self, img_ch=1, output_ch=1):
        super(IResNet, self).__init__()

        up_conv = True

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = ConvBlock(ch_in=img_ch, ch_out=64)
        self.conv1_extra = SingleConv(ch_in=64, ch_out=64)
        self.conv2 = ConvBlock(ch_in=64, ch_out=128)
        self.conv3 = ConvBlock(ch_in=128, ch_out=256)
        self.conv4 = ConvBlock(ch_in=256, ch_out=512)
        self.conv5 = ConvBlock(ch_in=512, ch_out=1024)

        self.resnet = ResidualBlock_mod(ch_in=1024)
        # self.resnet = ResidualBlock(ch_in=1024)

        self.up5 = UpConvBlock(ch_in=1024, ch_out=512, up_conv=up_conv)
        self.up_conv5 = ConvBlock(ch_in=1024, ch_out=512)
        self.up4 = UpConvBlock(ch_in=512, ch_out=256, up_conv=up_conv)
        self.up_conv4 = ConvBlock(ch_in=512, ch_out=256)
        self.up3 = UpConvBlock(ch_in=256, ch_out=128, up_conv=up_conv)
        self.up_conv3 = ConvBlock(ch_in=256, ch_out=128)
        self.up2 = UpConvBlock(ch_in=128, ch_out=64, up_conv=up_conv)
        self.up_conv2 = ConvBlock(ch_in=128, ch_out=64)

        self.conv_1x1 = nn.Conv2d(
            64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, input):
        # encoding path
        e1 = self.conv1(input)
        e1 = self.conv1_extra(e1)

        e2 = self.maxpool(e1)
        e2 = self.conv2(e2)

        e3 = self.maxpool(e2)
        e3 = self.conv3(e3)

        e4 = self.maxpool(e3)
        e4 = self.conv4(e4)

        e5 = self.maxpool(e4)
        e5 = self.conv5(e5)

        # ResNet Blocks
        r1 = self.resnet(e5)
        r2 = self.resnet(r1)
        r3 = self.resnet(r2)
        r4 = self.resnet(r3)
        r5 = self.resnet(r4)
        r6 = self.resnet(r5)
        r7 = self.resnet(r6)

        # decoding + concat path
        d5 = self.up5(r7)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.up_conv5(d5)

        d4 = self.up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.up_conv2(d2)

        d1 = self.conv_1x1(d2)

        # additional skip connection between input and output
        result = input + d1

        return result


class IResNet_mod(nn.Module):

    def __init__(self, img_ch=1, output_ch=1):
        super(IResNet_mod, self).__init__()

        up_conv = True

        self.conv1 = ConvBlock(ch_in=img_ch, ch_out=64)
        self.conv1_extra = SingleConv(ch_in=64, ch_out=64)
        self.conv2 = ConvBlock(ch_in=64, ch_out=128)
        self.conv3 = ConvBlock(ch_in=128, ch_out=256)
        self.conv4 = ConvBlock(ch_in=256, ch_out=512)
        self.conv5 = ConvBlock(ch_in=512, ch_out=1024)

        self.resnet = ResidualBlock_mod(ch_in=1024)
        # self.resnet = ResidualBlock(ch_in=1024)

        self.up5 = UpConvBlock(ch_in=1024, ch_out=512, up_conv=up_conv)
        self.up_conv5 = ConvBlock(ch_in=1024, ch_out=512)
        self.up4 = UpConvBlock(ch_in=512, ch_out=256, up_conv=up_conv)
        self.up_conv4 = ConvBlock(ch_in=512, ch_out=256)
        self.up3 = UpConvBlock(ch_in=256, ch_out=128, up_conv=up_conv)
        self.up_conv3 = ConvBlock(ch_in=256, ch_out=128)
        self.up2 = UpConvBlock(ch_in=128, ch_out=64, up_conv=up_conv)
        self.up_conv2 = ConvBlock(ch_in=128, ch_out=64)

        self.conv_1x1 = nn.Conv2d(
            64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, input):
        # encoding path
        e1 = self.conv1(input)
        e1 = self.conv1_extra(e1)

        e2 = self.conv2(e1)

        e3 = self.conv3(e2)

        e4 = self.conv4(e3)

        e5 = self.conv5(e4)

        # ResNet Blocks
        r1 = self.resnet(e5)
        r2 = self.resnet(r1)
        r3 = self.resnet(r2)
        r4 = self.resnet(r3)
        r5 = self.resnet(r4)
        r6 = self.resnet(r5)
        r7 = self.resnet(r6)

        # decoding + concat path
        d5 = self.up5(r7)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.up_conv5(d5)

        d4 = self.up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.up_conv2(d2)

        d1 = self.conv_1x1(d2)

        # additional skip connection between input and output
        result = input + d1

        return result
    
class SingleConvDropout(nn.Module):
    def __init__(self, ch_in, ch_out, p=0.2):
        super(SingleConvDropout, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=p),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvBlockDropout(nn.Module):
    def __init__(self, ch_in, ch_out, p=0.2):
        super(ConvBlockDropout, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=p),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=p),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConvBlockDropout(nn.Module):
    def __init__(self, ch_in, ch_out, up_conv, p=0.2):
        super(UpConvBlockDropout, self).__init__()

        if up_conv:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(
                    ch_in, ch_out, kernel_size=3, stride=2, padding=1, output_padding=1
                ),
                nn.InstanceNorm2d(ch_out),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=p),
            )

        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.InstanceNorm2d(ch_out),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=p),
            )

    def forward(self, x):
        x = self.up(x)
        return x


class ResidualBlock_modDropout(nn.Module):
    # ReflectionPad and InstanceNorm
    def __init__(self, ch_in, p=0.2):
        super(ResidualBlock_modDropout, self).__init__()

        self.block = nn.Sequential(
            # Pads the input tensor using the reflection of the input boundary
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch_in, ch_in, 3),
            nn.InstanceNorm2d(ch_in),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=p),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch_in, ch_in, 3),
            nn.InstanceNorm2d(ch_in),
            nn.Dropout2d(p=p),
        )
        self.relu = nn.Sequential(nn.ReLU(inplace=True), nn.Dropout2d(p=p))

    def forward(self, x):
        y = x + self.block(x)
        y1 = self.relu(y)
        return y1

class IResNetDropout(nn.Module):

    def __init__(self, img_ch=1, output_ch=1, p=0.15):
        super(IResNetDropout, self).__init__()

        up_conv = True

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = ConvBlock(ch_in=img_ch, ch_out=64)
        self.conv1_extra = ConvBlock(ch_in=64, ch_out=64)
        self.conv2 = ConvBlockDropout(ch_in=64, ch_out=128, p=p)
        self.conv3 = ConvBlockDropout(ch_in=128, ch_out=256, p=p)
        self.conv4 = ConvBlockDropout(ch_in=256, ch_out=512, p=p)
        self.conv5 = ConvBlockDropout(ch_in=512, ch_out=1024, p=p)

        self.resnet = ResidualBlock_modDropout(ch_in=1024, p=p)
        # self.resnet = ResidualBlock(ch_in=1024)

        self.up5 = UpConvBlockDropout(ch_in=1024, ch_out=512, up_conv=up_conv, p=p)
        self.up_conv5 = ConvBlockDropout(ch_in=1024, ch_out=512, p=p)
        self.up4 = UpConvBlockDropout(ch_in=512, ch_out=256, up_conv=up_conv, p=p)
        self.up_conv4 = ConvBlockDropout(ch_in=512, ch_out=256, p=p)
        self.up3 = UpConvBlockDropout(ch_in=256, ch_out=128, up_conv=up_conv, p=p)
        self.up_conv3 = ConvBlockDropout(ch_in=256, ch_out=128, p=p)
        self.up2 = UpConvBlock(ch_in=128, ch_out=64, up_conv=up_conv)
        self.up_conv2 = ConvBlock(ch_in=128, ch_out=64)

        self.conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, input):
        # encoding path
        e1 = self.conv1(input)
        e1 = self.conv1_extra(e1)

        e2 = self.maxpool(e1)
        e2 = self.conv2(e2)

        e3 = self.maxpool(e2)
        e3 = self.conv3(e3)

        e4 = self.maxpool(e3)
        e4 = self.conv4(e4)

        e5 = self.maxpool(e4)
        e5 = self.conv5(e5)

        # ResNet Blocks
        r1 = self.resnet(e5)
        r2 = self.resnet(r1)
        r3 = self.resnet(r2)
        r4 = self.resnet(r3)
        r5 = self.resnet(r4)
        r6 = self.resnet(r5)
        r7 = self.resnet(r6)

        # decoding + concat path
        d5 = self.up5(r7)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.up_conv5(d5)

        d4 = self.up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.up_conv2(d2)

        d1 = self.conv_1x1(d2)

        # additional skip connection between input and output
        result = input + d1

        return result
    

def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)


def unflatten_like(vector, likeTensorList):
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    #    shaped like likeTensorList
    outList = []
    i = 0
    for tensor in likeTensorList:
        # n = module._parameters[name].numel()
        n = tensor.numel()
        outList.append(vector[:, i : i + n].view(tensor.shape))
        i += n
    return outList

def swag_parameters(module, params):
    """
    A helper function to recursively replace nn.Parameters with SWAG buffers.
    This is adapted directly from the official SWAG implementation.
    """
    for name in list(module._parameters.keys()):
        if module._parameters[name] is None:
            continue
        data = module._parameters[name].data
        module._parameters.pop(name)
        
        # Register buffers for the SWAG statistics
        module.register_buffer(f"{name}_mean", data.new_zeros(data.size()))
        module.register_buffer(f"{name}_sq_mean", data.new_zeros(data.size()))
        module.register_buffer(f"{name}_cov_mat_sqrt", data.new_empty((0, data.numel())))
        
        params.append((module, name))


class SWAG(nn.Module):
    """
    SWAG Model Wrapper.

    This implementation is a close adaptation of the official SWAG repository's code,
    modified for a post-hoc workflow where model checkpoints are loaded from disk.
    """
    def __init__(self, base_model_cls, swag_checkpoint_paths, swag_cov_mat_rank, **kwargs):
        super().__init__()

        self.params = []
        self.base_model_cls = base_model_cls
        self.base_model_kwargs = kwargs
        self.base_model = self.base_model_cls(**self.base_model_kwargs)
        
        # This function traverses the model and sets up the buffers for SWAG statistics
        self.base_model.apply(lambda module: swag_parameters(module, self.params))

        # --- Post-Hoc Collection from Checkpoints ---
        self._post_hoc_collect(swag_checkpoint_paths, swag_cov_mat_rank)
        
    def forward(self, *args, **kwargs):
        """A forward pass uses the weights currently loaded in the base_model."""
        return self.base_model(*args, **kwargs)

    def _post_hoc_collect(self, checkpoint_paths, max_num_models):
        """
        Performs a post-hoc collection of SWAG statistics by iterating through
        saved model checkpoints.
        """
        # Load all state dicts from the provided paths
        state_dicts = [torch.load(path) for path in checkpoint_paths]
        
        num_models = 0
        
        for i, state_dict in enumerate(tqdm(state_dicts, desc="Computing SWAG Statistics")):
            # Create a temporary base model instance and load the state dict
            temp_model = self.base_model_cls(**self.base_model_kwargs)
            temp_model.load_state_dict(state_dict)

            # Use the same logic as the online 'collect_model' method
            for (module, name), base_param in zip(self.params, temp_model.parameters()):
                mean = module.__getattr__(f"{name}_mean")
                sq_mean = module.__getattr__(f"{name}_sq_mean")

                # Update first moment
                mean.mul_(num_models / (num_models + 1.0)).add_(base_param.data / (num_models + 1.0))

                # Update second moment
                sq_mean.mul_(num_models / (num_models + 1.0)).add_(base_param.data**2 / (num_models + 1.0))

                # --- CORRECTED: Update deviation using the new running mean ---
                dev = (base_param.data - mean)
                
                cov_mat_sqrt = module.__getattr__(f"{name}_cov_mat_sqrt")
                cov_mat_sqrt = torch.cat((cov_mat_sqrt, dev.view(1, -1)), dim=0)

                # Keep only the last K deviations
                if cov_mat_sqrt.shape[0] > max_num_models:
                    cov_mat_sqrt = cov_mat_sqrt[1:, :]
                
                module.__setattr__(f"{name}_cov_mat_sqrt", cov_mat_sqrt)
            
            num_models += 1

        # Store the final count of models used
        self.register_buffer("n_models", torch.tensor(num_models, dtype=torch.long))

    def sample(self, scale=1.0,seed=None):
        """
        Samples a new set of weights from the SWAG posterior and loads them
        into the base model for a forward pass. This is an adaptation of `sample_fullrank`.
        """
        if seed is not None:
            torch.manual_seed(seed)

        # 1. Flatten all SWAG statistics into vectors
        mean_list = [m.__getattr__(f"{n}_mean") for m, n in self.params]
        sq_mean_list = [m.__getattr__(f"{n}_sq_mean") for m, n in self.params]
        cov_mat_sqrt_list = [m.__getattr__(f"{n}_cov_mat_sqrt") for m, n in self.params]

        mean_vec = flatten(mean_list)
        sq_mean_vec = flatten(sq_mean_list)
        
        # 2. Draw sample from the diagonal part
        var_vec = torch.clamp(sq_mean_vec - mean_vec**2, 1e-30)
        diag_sample = torch.sqrt(var_vec) * torch.randn_like(var_vec)

        # 3. Draw sample from the low-rank part
        # Concatenate column-wise for the low-rank matrix
        cov_mat_sqrt = torch.cat(cov_mat_sqrt_list, dim=1) 
        
        z = torch.randn(cov_mat_sqrt.shape[0], device=cov_mat_sqrt.device)
        low_rank_sample = (cov_mat_sqrt.t() @ z)
        
        num_models = self.n_models.item()
        if num_models > 1:
            low_rank_sample /= math.sqrt(num_models - 1)
        
        # 4. Combine samples
        rand_sample = scale * (diag_sample + low_rank_sample) / math.sqrt(2.0)
        w_sample = mean_vec + rand_sample

        # 5. Load the sampled weights back into the model
        # The unflatten_like function expects a batch dimension
        w_sample = w_sample.unsqueeze(0)
        samples_list = unflatten_like(w_sample, mean_list)

        for (module, name), sample in zip(self.params, samples_list):
            module.register_parameter(name, nn.Parameter(sample))



# =================================================================
# Helper Functions for Probability Distributions
# =================================================================

def log_gaussian_prob(x, mu, rho):
    """
    Calculates the log probability of a value 'x' under a Gaussian distribution.
    The standard deviation (sigma) is derived from 'rho' to ensure it's positive.
    sigma = log(1 + exp(rho))
    """
    sigma = torch.nn.functional.softplus(rho)  # Ensures sigma is positive
    # log N(x | mu, sigma^2)
    return distributions.Normal(mu, sigma).log_prob(x)

def log_scale_mixture_gaussian_prob(x, pi, mu, sigma1, sigma2):
    """
    Calculates the log probability of a value 'x' under a scale mixture of two
    Gaussian distributions.
    p(x) = pi * N(x | mu, sigma1^2) + (1 - pi) * N(x | mu, sigma2^2)
    Uses the log-sum-exp trick for numerical stability.
    """
    # Log probabilities for each component of the mixture
    log_prob1 = distributions.Normal(mu, sigma1).log_prob(x)
    log_prob2 = distributions.Normal(mu, sigma2).log_prob(x)

    # Combine using log-sum-exp: log(exp(a) + exp(b))
    log_mix_prob = torch.logsumexp(
        torch.stack([torch.log(pi) + log_prob1, torch.log(1 - pi) + log_prob2]),
        dim=0
    )
    return log_mix_prob

# =================================================================
# Bayes by Backprop (BBB) Implementation
# =================================================================

class BayesianLayer(nn.Module):
    """
    Base class for a Bayesian layer. It defines the scale mixture Gaussian prior
    and the logic for calculating the KL divergence term via sampling.
    The KL divergence is approximated as log(q(w|D)) - log(p(w)).
    """
    def __init__(self, prior_pi=0.5, prior_sigma1=1.0, prior_sigma2=0.001, prior_mu=0.0):
        super().__init__()
        # Register prior parameters as buffers to ensure they are moved to the correct device
        self.register_buffer('prior_pi', torch.tensor(float(prior_pi)))
        self.register_buffer('prior_sigma1', torch.tensor(float(prior_sigma1)))
        self.register_buffer('prior_sigma2', torch.tensor(float(prior_sigma2)))
        self.register_buffer('prior_mu', torch.tensor(float(prior_mu)))

        # Placeholders for the log probabilities of the sampled weights
        self.log_q = 0  # Log probability of the weights under the posterior q
        self.log_p = 0  # Log probability of the weights under the prior p

    def log_prob_prior(self, w):
        """Calculates the log probability of the weights 'w' under the mixture prior."""
        return log_scale_mixture_gaussian_prob(w, self.prior_pi, self.prior_mu, self.prior_sigma1, self.prior_sigma2)

    def log_prob_posterior(self, w, mu, rho):
        """Calculates the log probability of the weights 'w' under the posterior."""
        return log_gaussian_prob(w, mu, rho)

    def get_kl_divergence_term(self):
        """Returns the calculated KL divergence term: log(q) - log(p)."""
        return self.log_q - self.log_p

    def extra_repr(self):
        return f'prior_pi={self.prior_pi.item():.2f}, prior_mu={self.prior_mu.item()}, prior_sigma1={self.prior_sigma1.item():.4f}, prior_sigma2={self.prior_sigma2.item():.4f}'


class BayesianConv2d(BayesianLayer):
    """
    Bayesian Convolutional Layer using the reparameterization trick and sampling-based KL.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True,
                 prior_pi=0.5, prior_sigma1=1.0, prior_sigma2=0.001, prior_mu=0.0):
        super().__init__(prior_pi, prior_sigma1, prior_sigma2, prior_mu)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.has_bias = bias

        # Posterior parameters for weights (mu and rho for the Gaussian)
        self.weight_mu = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.weight_rho = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))

        if self.has_bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_channels))
            self.bias_rho = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize posterior means similar to a standard Conv2d layer
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        # Initialize rho to a small negative value for low initial variance
        nn.init.constant_(self.weight_rho, -5.0)
        if self.has_bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_mu, -bound, bound)
            nn.init.constant_(self.bias_rho, -5.0)

    def forward(self, x):
        # --- Sample weights and bias from their posteriors using the reparameterization trick ---
        # 1. Get standard deviation from rho
        weight_sigma = torch.nn.functional.softplus(self.weight_rho)
        # 2. Sample from a standard normal
        epsilon_w = torch.randn_like(weight_sigma)
        # 3. Compute the sampled weight
        weight = self.weight_mu + weight_sigma * epsilon_w

        if self.has_bias:
            bias_sigma = torch.nn.functional.softplus(self.bias_rho)
            epsilon_b = torch.randn_like(bias_sigma)
            bias = self.bias_mu + bias_sigma * epsilon_b
        else:
            bias = None

        # --- Calculate log probabilities for the KL term ---
        # log q(w|D)
        self.log_q = self.log_prob_posterior(weight, self.weight_mu, self.weight_rho).sum()
        # log p(w)
        self.log_p = self.log_prob_prior(weight).sum()

        if self.has_bias:
            self.log_q += self.log_prob_posterior(bias, self.bias_mu, self.bias_rho).sum()
            self.log_p += self.log_prob_prior(bias).sum()

        # --- Perform the convolution with the sampled weights ---
        return F.conv2d(x, weight, bias, self.stride, self.padding)

    def extra_repr(self):
        base = f'in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, bias={self.has_bias}'
        prior_info = super().extra_repr()
        return f'{base}, {prior_info}'


# --- Bayesian versions of the U-Net building blocks ---

class BayesianConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, **prior_kwargs):
        super(BayesianConvBlock, self).__init__()
        self.conv = nn.Sequential(
            BayesianConv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True, **prior_kwargs),
            nn.InstanceNorm2d(ch_out),
            nn.ReLU(inplace=True),
            BayesianConv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True, **prior_kwargs),
            nn.InstanceNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class BayesianUpConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, up_conv, **prior_kwargs):
        super(BayesianUpConvBlock, self).__init__()
        # Using a deterministic ConvTranspose2d for simplicity
        # The subsequent convolutional blocks will be Bayesian.
        if up_conv:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(ch_out),
                nn.ReLU(inplace=True)
            )
        else:
            # Upsample is deterministic, but the following conv should be Bayesian
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                BayesianConv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True, **prior_kwargs),
                nn.InstanceNorm2d(ch_out),
                nn.ReLU(inplace=True)
            )
    def forward(self, x):
        return self.up(x)

class BayesianResidualBlock_mod(nn.Module):
    def __init__(self, ch_in, **prior_kwargs):
        super(BayesianResidualBlock_mod, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            BayesianConv2d(ch_in, ch_in, 3, **prior_kwargs),
            nn.InstanceNorm2d(ch_in),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            BayesianConv2d(ch_in, ch_in, 3, **prior_kwargs),
            nn.InstanceNorm2d(ch_in)
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        y = x + self.block(x)
        y1 = self.relu(y)
        return y1


class IResNetBBB(nn.Module):
    def __init__(self, img_ch=1, output_ch=1, **prior_kwargs):
        super(IResNetBBB, self).__init__()

        # This architecture keeps the first and last few layers deterministic,
        up_conv = True

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Deterministic Entry Layers ---
        self.conv1 = ConvBlock(ch_in=img_ch, ch_out=64)
        self.conv1_extra = SingleConv(ch_in=64, ch_out=64)

        # --- Bayesian Encoding Layers ---
        self.conv2 = BayesianConvBlock(ch_in=64, ch_out=128, **prior_kwargs)
        self.conv3 = BayesianConvBlock(ch_in=128, ch_out=256, **prior_kwargs)
        self.conv4 = BayesianConvBlock(ch_in=256, ch_out=512, **prior_kwargs)
        self.conv5 = BayesianConvBlock(ch_in=512, ch_out=1024, **prior_kwargs)

        # --- Bayesian Bottleneck ---
        self.resnet = BayesianResidualBlock_mod(ch_in=1024, **prior_kwargs)

        # --- Bayesian Decoding Layers ---
        self.up5 = BayesianUpConvBlock(ch_in=1024, ch_out=512, up_conv=up_conv, **prior_kwargs)
        self.up_conv5 = BayesianConvBlock(ch_in=1024, ch_out=512, **prior_kwargs)
        self.up4 = BayesianUpConvBlock(ch_in=512, ch_out=256, up_conv=up_conv, **prior_kwargs)
        self.up_conv4 = BayesianConvBlock(ch_in=512, ch_out=256, **prior_kwargs)
        self.up3 = BayesianUpConvBlock(ch_in=256, ch_out=128, up_conv=up_conv, **prior_kwargs)
        self.up_conv3 = BayesianConvBlock(ch_in=256, ch_out=128, **prior_kwargs)

        # --- Deterministic Exit Layers ---
        self.up2 = UpConvBlock(ch_in=128, ch_out=64, up_conv=up_conv)
        self.up_conv2 = ConvBlock(ch_in=128, ch_out=64)
        self.conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, input):
        # encoding path
        e1 = self.conv1(input)
        e1 = self.conv1_extra(e1)

        e2 = self.maxpool(e1)
        e2 = self.conv2(e2)

        e3 = self.maxpool(e2)
        e3 = self.conv3(e3)

        e4 = self.maxpool(e3)
        e4 = self.conv4(e4)

        e5 = self.maxpool(e4)
        e5 = self.conv5(e5)

        # ResNet Blocks
        r1 = self.resnet(e5)
        r2 = self.resnet(r1)
        r3 = self.resnet(r2)
        r4 = self.resnet(r3)
        r5 = self.resnet(r4)
        r6 = self.resnet(r5)
        r7 = self.resnet(r6)

        # decoding + concat path
        d5 = self.up5(r7)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.up_conv5(d5)

        d4 = self.up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.up_conv2(d2)

        d1 = self.conv_1x1(d2)

        # additional skip connection between input and output
        result = input + d1

        return result

    def kl_divergence(self):
        """
        Calculates the total KL divergence term for the model loss.
        It sums the 'log(q) - log(p)' from all BayesianLayer modules.
        """
        total_kl_term = 0
        for module in self.modules():
            if isinstance(module, BayesianLayer):
                # The KL term of each layer is computed during the forward pass.
                # Here we just sum them up.
                total_kl_term += module.get_kl_divergence_term()
        return total_kl_term


class IResNetEvidential(nn.Module):

    def __init__(self, img_ch=1):
        super(IResNetEvidential, self).__init__()

        up_conv = True

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = ConvBlock(ch_in=img_ch, ch_out=64)
        self.conv1_extra = SingleConv(ch_in=64, ch_out=64)
        self.conv2 = ConvBlock(ch_in=64, ch_out=128)
        self.conv3 = ConvBlock(ch_in=128, ch_out=256)
        self.conv4 = ConvBlock(ch_in=256, ch_out=512)
        self.conv5 = ConvBlock(ch_in=512, ch_out=1024)

        self.resnet = ResidualBlock_mod(ch_in=1024)
        # self.resnet = ResidualBlock(ch_in=1024)

        self.up5 = UpConvBlock(ch_in=1024, ch_out=512, up_conv=up_conv)
        self.up_conv5 = ConvBlock(ch_in=1024, ch_out=512)
        self.up4 = UpConvBlock(ch_in=512, ch_out=256, up_conv=up_conv)
        self.up_conv4 = ConvBlock(ch_in=512, ch_out=256)
        self.up3 = UpConvBlock(ch_in=256, ch_out=128, up_conv=up_conv)
        self.up_conv3 = ConvBlock(ch_in=256, ch_out=128)
        self.up2 = UpConvBlock(ch_in=128, ch_out=64, up_conv=up_conv)
        self.up_conv2 = ConvBlock(ch_in=128, ch_out=64)

        self.conv_1x1 = nn.Conv2d(64, 4, kernel_size=1, stride=1, padding=0)

    def forward(self, input):
        # encoding path
        e1 = self.conv1(input)
        e1 = self.conv1_extra(e1)

        e2 = self.maxpool(e1)
        e2 = self.conv2(e2)

        e3 = self.maxpool(e2)
        e3 = self.conv3(e3)

        e4 = self.maxpool(e3)
        e4 = self.conv4(e4)

        e5 = self.maxpool(e4)
        e5 = self.conv5(e5)

        # ResNet Blocks
        r1 = self.resnet(e5)
        r2 = self.resnet(r1)
        r3 = self.resnet(r2)
        r4 = self.resnet(r3)
        r5 = self.resnet(r4)
        r6 = self.resnet(r5)
        r7 = self.resnet(r6)

        # decoding + concat path
        d5 = self.up5(r7)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.up_conv5(d5)

        d4 = self.up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.up_conv2(d2)

        d1 = self.conv_1x1(d2)

        # additional skip connection between input and output channel 1
        gamma = input + d1[:, 0:1, :, :]

        # softplus the other channels to ensure positivity (and add 1 to alpha)
        # and an additional small bit, in case there is underflow in softplus
        nu = F.softplus(d1[:, 1:2, :, :]) + 1e-6
        alpha = F.softplus(d1[:, 2:3, :, :]) + 1.0 + 1e-6
        beta = F.softplus(d1[:, 3:4, :, :]) + 1e-6

        return gamma, nu, alpha, beta
    

class IResNetAuxiliary(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(IResNetAuxiliary, self).__init__()

        up_conv = True

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = ConvBlock(ch_in=img_ch, ch_out=64)
        self.conv1_extra = SingleConv(ch_in=64, ch_out=64)
        self.conv2 = ConvBlock(ch_in=64, ch_out=128)
        self.conv3 = ConvBlock(ch_in=128, ch_out=256)
        self.conv4 = ConvBlock(ch_in=256, ch_out=512)
        self.conv5 = ConvBlock(ch_in=512, ch_out=1024)

        self.resnet = ResidualBlock_mod(ch_in=1024)
        # self.resnet = ResidualBlock(ch_in=1024)

        self.up5 = UpConvBlock(ch_in=1024, ch_out=512, up_conv=up_conv)
        self.up_conv5 = ConvBlock(ch_in=1024, ch_out=512)
        self.up4 = UpConvBlock(ch_in=512, ch_out=256, up_conv=up_conv)
        self.up_conv4 = ConvBlock(ch_in=512, ch_out=256)
        self.up3 = UpConvBlock(ch_in=256, ch_out=128, up_conv=up_conv)
        self.up_conv3 = ConvBlock(ch_in=256, ch_out=128)
        self.up2 = UpConvBlock(ch_in=128, ch_out=64, up_conv=up_conv)
        self.up_conv2 = ConvBlock(ch_in=128, ch_out=64)

        self.conv_1x1 = nn.Conv2d(
            64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, input):
        # encoding path
        e1 = self.conv1(input)
        e1 = self.conv1_extra(e1)

        e2 = self.maxpool(e1)
        e2 = self.conv2(e2)

        e3 = self.maxpool(e2)
        e3 = self.conv3(e3)

        e4 = self.maxpool(e3)
        e4 = self.conv4(e4)

        e5 = self.maxpool(e4)
        e5 = self.conv5(e5)

        # ResNet Blocks
        r1 = self.resnet(e5)
        r2 = self.resnet(r1)
        r3 = self.resnet(r2)
        r4 = self.resnet(r3)
        r5 = self.resnet(r4)
        r6 = self.resnet(r5)
        r7 = self.resnet(r6)

        # decoding + concat path
        d5 = self.up5(r7)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.up_conv5(d5)

        d4 = self.up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.up_conv2(d2)

        d1 = self.conv_1x1(d2)

        # No skip connect

        return d1