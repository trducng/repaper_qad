# Simple UNet model that can serve as transformer
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VGGFeatures(nn.Module):
    """Pre-trained VGG model"""

    def __init__(self, pretrained=False, requires_grad=False):
        """Initialize the object"""
        super(VGGFeatures, self).__init__()
        features = models.vgg16_bn(pretrained=pretrained).features

        self.slice1, self.slice2 = nn.Sequential(), nn.Sequential()
        self.slice3, self.slice4 = nn.Sequential(), nn.Sequential()
        for idx in range(6):
            self.slice1.add_module(str(idx), features[idx])
        for idx in range(7, 13):
            self.slice2.add_module(str(idx), features[idx])
        for idx in range(14, 23):
            self.slice3.add_module(str(idx), features[idx])
        for idx in range(23, 33):
            self.slice4.add_module(str(idx), features[idx])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input_x):
        """Perform the forward pass"""
        h = self.slice1(input_x)
        relu1 = h
        h = self.slice2(h)
        relu2 = h
        h = self.slice3(h)
        relu3 = h
        h = self.slice4(h)
        relu4 = h

        vgg_outputs = namedtuple("Outputs", ["relu1", "relu2", "relu3", "relu4"])
        out = vgg_outputs(relu1, relu2, relu3, relu4)
        return out


class ResBlock(nn.Module):
    """Residual block"""

    def __init__(self, in_channels, out_channels):
        """Initialize the object"""
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, input_x):
        """Perform the forward pass"""
        h = self.conv(input_x)
        h += self.identity(input_x)
        out = F.relu(h)
        return out


class UNet(nn.Module):
    """UNet model based on VGG architecture, usually used for semantic segmentation"""

    def __init__(self, pretrained=False):
        """Initialize the object"""
        super(UNet, self).__init__()

        # encoder
        self.basenet = VGGFeatures(pretrained=pretrained, requires_grad=True)

        # decoder
        self.upconv1 = ResBlock(in_channels=256, out_channels=64)
        self.upconv2 = ResBlock(in_channels=128 + 64, out_channels=32)
        self.upconv3 = ResBlock(in_channels=64 + 32, out_channels=16)

        self.classifier = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=1, padding=0),
        )

    def forward(self, input_x):
        """Perform forward pass"""

        # encoder
        base = self.basenet(input_x)

        # decoder
        h = self.upconv1(base[2])

        h = F.interpolate(
            h, size=base[1].size()[2:], mode="bilinear", align_corners=False
        )
        h = torch.cat([h, base[1]], dim=1)
        h = self.upconv2(h)

        h = F.interpolate(
            h, size=base[0].size()[2:], mode="bilinear", align_corners=False
        )
        h = torch.cat([h, base[0]], dim=1)
        h = self.upconv3(h)

        out = self.classifier(h)
        return torch.sigmoid(out)


if __name__ == "__main__":
    input_x = torch.randn(1, 3, 512, 512)
    unet = UNet(pretrained=False)
    print(unet(input_x).shape)
