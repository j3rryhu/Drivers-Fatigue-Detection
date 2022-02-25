import torch as t
import torch.nn as nn
import torch.nn.functional as F


class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BottleNeck, self).__init__()
        first_stride = 0
        if in_channels != out_channels and in_channels != 64:
            first_stride = 2
        elif in_channels != out_channels and in_channels == 64:
            first_stride = 1
        elif in_channels == out_channels:
            first_stride = 1
        self.residual_func = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, 1, stride=first_stride, bias=False),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//4, out_channels//4, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//4, out_channels, 1, stride=1, bias=False, dilation=2),
            nn.BatchNorm2d(out_channels),
        )

        if in_channels != out_channels and in_channels == 64:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=1),
                                          nn.BatchNorm2d(out_channels))
        elif in_channels != out_channels and in_channels != 64:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=2),
                                          nn.BatchNorm2d(out_channels))
        elif in_channels == out_channels:
            self.shortcut = nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.residual_func(x)
        identity = self.shortcut(identity)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, in_channels=3):
        super(ResNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3), padding=1, stride=2)
        )
        self.layers = nn.Sequential()
        for i in range(0, 16):
            if 0 <= i <= 2:
                if i == 0:
                    self.layers.add_module("conv{}".format(i+1), BottleNeck(64, 256))
                else:
                    self.layers.add_module("conv{}".format(i + 1), BottleNeck(256, 256))
            elif 3<=i<=6:
                if i == 3:
                    self.layers.add_module("conv{}".format(i + 1), BottleNeck(256, 512))
                else:
                    self.layers.add_module("conv{}".format(i + 1), BottleNeck(512, 512))
            elif 7<=i<=12:
                if i == 7:
                    self.layers.add_module("conv{}".format(i + 1), BottleNeck(512, 1024))
                else:
                    self.layers.add_module("conv{}".format(i + 1), BottleNeck(1024, 1024))
            elif 13<=i<=15:
                if i == 13:
                    self.layers.add_module("conv{}".format(i + 1), BottleNeck(1024, 2048))
                else:
                    self.layers.add_module("conv{}".format(i + 1), BottleNeck(2048, 2048))
        self.fc = nn.Linear(2048*3*4, 136)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.layers(out)
        out = out.view(-1, 2048*3*4)
        out = self.fc(out)
        return out
