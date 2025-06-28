import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math


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

        self.conv1 = ConvBlock(ch_in=img_ch, ch_out=64, p=p)
        self.conv1_extra = ConvBlock(ch_in=64, ch_out=64, p=p)
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
        self.up2 = UpConvBlock(ch_in=128, ch_out=64, up_conv=up_conv, p=p)
        self.up_conv2 = ConvBlock(ch_in=128, ch_out=64, p=p)

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
    


# =================================================================
# Bayes by Backprop Implementation
# =================================================================

class BayesianLayer(nn.Module):
    """
    Base class for a Bayesian layer. It is used to calculate the KL divergence.
    """
    def __init__(self, prior_mu=0, prior_sigma=0.10):
        super().__init__()
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        
        # We need to register these as buffers so that they are moved to the correct device
        self.register_buffer('prior_mu_tensor', torch.tensor(prior_mu))
        self.register_buffer('prior_sigma_tensor', torch.tensor(prior_sigma))

        self.kl_divergence = 0

    def _kl_divergence(self, mu, sigma):
        """
        Calculates the KL divergence between a posterior Gaussian and a prior Gaussian.
        q(w) = N(mu, sigma)
        p(w) = N(self.prior_mu, self.prior_sigma)
        """
        # Ensure sigma is positive
        sigma = torch.log(1 + torch.exp(sigma))
        
        # The KL divergence between two univariate Gaussians
        kl = (torch.log(self.prior_sigma_tensor / sigma)
              + (sigma**2 + (mu - self.prior_mu_tensor)**2) / (2 * self.prior_sigma_tensor**2)
              - 0.5)
        return kl.sum()

class BayesianConv2d(BayesianLayer):
    """
    Bayesian Convolutional Layer using the reparameterization trick.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.has_bias = bias

        # Posterior parameters for weights
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
        # Initialization similar to nn.Conv2d
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        # Initialize rho to a small value for low initial variance
        nn.init.constant_(self.weight_rho, -3.0) 
        if self.has_bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_mu, -bound, bound)
            nn.init.constant_(self.bias_rho, -3.0)

    def forward(self, x):
        # Sample weights and bias from their posteriors using the reparameterization trick
        # 1. Get standard deviation from rho
        weight_sigma = torch.log(1 + torch.exp(self.weight_rho))
        # 2. Sample from a standard normal
        epsilon_w = torch.randn_like(weight_sigma)
        # 3. Compute the sampled weight
        weight = self.weight_mu + weight_sigma * epsilon_w

        if self.has_bias:
            bias_sigma = torch.log(1 + torch.exp(self.bias_rho))
            epsilon_b = torch.randn_like(bias_sigma)
            bias = self.bias_mu + bias_sigma * epsilon_b
        else:
            bias = None

        # Calculate KL divergence for this forward pass and store it
        self.kl_divergence = self._kl_divergence(self.weight_mu, self.weight_rho)
        if self.has_bias:
            self.kl_divergence += self._kl_divergence(self.bias_mu, self.bias_rho)

        # Perform the convolution with the sampled weights
        return F.conv2d(x, weight, bias, self.stride, self.padding)

# --- Bayesian versions of the building blocks ---

class BayesianSingleConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(BayesianSingleConv, self).__init__()
        self.conv = nn.Sequential(
            BayesianConv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class BayesianConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(BayesianConvBlock, self).__init__()
        self.conv = nn.Sequential(
            BayesianConv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(ch_out),
            nn.ReLU(inplace=True),
            BayesianConv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class BayesianUpConvBlock(nn.Module):
    # NOTE: Using deterministic ConvTranspose2d as it's typically not a major source of uncertainty
    # and Bayesian ConvTranspose2d can be complex to implement correctly.
    def __init__(self, ch_in, ch_out, up_conv):
        super(BayesianUpConvBlock, self).__init__()
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
                BayesianConv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.InstanceNorm2d(ch_out),
                nn.ReLU(inplace=True)
            )
    def forward(self, x):
        return self.up(x)
        
class BayesianResidualBlock_mod(nn.Module):
    def __init__(self, ch_in):
        super(BayesianResidualBlock_mod, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            BayesianConv2d(ch_in, ch_in, 3),
            nn.InstanceNorm2d(ch_in),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            BayesianConv2d(ch_in, ch_in, 3),
            nn.InstanceNorm2d(ch_in)
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        y = x + self.block(x)
        y1 = self.relu(y)
        return y1


class IResNetBBB(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(IResNetBBB, self).__init__()

        up_conv = True

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = BayesianConvBlock(ch_in=img_ch, ch_out=64)
        self.conv1_extra = BayesianSingleConv(ch_in=64, ch_out=64)
        self.conv2 = BayesianConvBlock(ch_in=64, ch_out=128)
        self.conv3 = BayesianConvBlock(ch_in=128, ch_out=256)
        self.conv4 = BayesianConvBlock(ch_in=256, ch_out=512)
        self.conv5 = BayesianConvBlock(ch_in=512, ch_out=1024)

        self.resnet = BayesianResidualBlock_mod(ch_in=1024)

        self.up5 = BayesianUpConvBlock(ch_in=1024, ch_out=512, up_conv=up_conv)
        self.up_conv5 = BayesianConvBlock(ch_in=1024, ch_out=512)
        self.up4 = BayesianUpConvBlock(ch_in=512, ch_out=256, up_conv=up_conv)
        self.up_conv4 = BayesianConvBlock(ch_in=512, ch_out=256)
        self.up3 = BayesianUpConvBlock(ch_in=256, ch_out=128, up_conv=up_conv)
        self.up_conv3 = BayesianConvBlock(ch_in=256, ch_out=128)
        self.up2 = BayesianUpConvBlock(ch_in=128, ch_out=64, up_conv=up_conv)
        self.up_conv2 = BayesianConvBlock(ch_in=128, ch_out=64)

        # Final layer is often kept deterministic, but can be Bayesian as well
        self.conv_1x1 = BayesianConv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

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
        Calculates the total KL divergence of the model by summing the KL
        of all `BayesianLayer` modules.
        """
        total_kl = 0
        for module in self.modules():
            if isinstance(module, BayesianLayer):
                # The KL divergence of each layer is computed and stored during the forward pass.
                # Here we just sum them up.
                total_kl += module.kl_divergence
        return total_kl