import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBn2d(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)):
        super(ConvBn2d, self).__init__(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                                       nn.BatchNorm2d(out_channels))


class ConvBnReLU2d(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)):
        super(ConvBnReLU2d, self).__init__(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                                           nn.BatchNorm2d(out_channels), nn.ReLU())


class PoolResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super(PoolResBlock, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), ceil_mode=True)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.1)
        self.branch0 = ConvBn2d(in_channels, out_channels, kernel_size=(1, 1))
        self.branch1 = nn.Sequential(
            ConvBnReLU2d(in_channels, out_channels // 4, kernel_size=(kernel_size, 1), padding=(kernel_size // 2, 0)),
            ConvBnReLU2d(out_channels // 4, out_channels // 4, kernel_size=(1, kernel_size),
                         padding=(0, kernel_size // 2)))
        self.branch2 = nn.Sequential(
            ConvBnReLU2d(in_channels, out_channels // 4, kernel_size=(1, kernel_size), padding=(0, kernel_size // 2)),
            ConvBnReLU2d(out_channels // 4, out_channels // 4, kernel_size=(kernel_size, 1),
                         padding=(kernel_size // 2, 0)))
        self.merge12 = ConvBn2d(out_channels // 4, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        out = self.pool(x)
        branch0 = self.branch0(out)
        branch1 = self.branch1(out)
        branch2 = self.branch2(out)
        out = branch0 + self.merge12(branch1 + branch2)
        out = self.relu(out)
        out = self.drop(out)
        return out


class DownResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, new_resnet=True):
        super(DownResBlock, self).__init__()
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.1)
        if new_resnet:
            self.branch0 = nn.Sequential(nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
                                         ConvBn2d(in_channels, out_channels, kernel_size=(1, 1)))
        else:
            self.branch0 = ConvBn2d(in_channels, out_channels, kernel_size=(1, 1), stride=(2, 2))
        self.branch1 = nn.Sequential(
            ConvBnReLU2d(in_channels, out_channels // 4, kernel_size=(kernel_size, 1), padding=(kernel_size // 2, 0),
                         stride=(2, 2)),
            ConvBnReLU2d(out_channels // 4, out_channels // 4, kernel_size=(1, kernel_size),
                         padding=(0, kernel_size // 2)))
        self.branch2 = nn.Sequential(
            ConvBnReLU2d(in_channels, out_channels // 4, kernel_size=(1, kernel_size), padding=(0, kernel_size // 2),
                         stride=(2, 2)),
            ConvBnReLU2d(out_channels // 4, out_channels // 4, kernel_size=(kernel_size, 1),
                         padding=(kernel_size // 2, 0)))
        self.merge12 = ConvBn2d(out_channels // 4, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        branch0 = self.branch0(x)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        out = branch0 + self.merge12(branch1 + branch2)
        out = self.relu(out)
        out = self.drop(out)
        return out


class ClassifyTailBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ClassifyTailBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class LargeKernelClassifierNet(nn.Module):

    def __init__(self, new_resnet=False):
        super(LargeKernelClassifierNet, self).__init__()
        self.databn = nn.BatchNorm2d(3)  # output 256x128
        if new_resnet:
            self.conv1 = nn.Sequential(ConvBnReLU2d(3, 4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                                       ConvBnReLU2d(4, 4, kernel_size=(3, 3), padding=(1, 1)),
                                       ConvBnReLU2d(4, 8, kernel_size=(3, 3), padding=(1, 1)),
                                       nn.Dropout(p=0.1))  # output 128x64
        else:
            self.conv1 = nn.Sequential(ConvBnReLU2d(3, 8, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)),
                                       nn.Dropout(p=0.1))  # output 128x64
        self.pool2 = PoolResBlock(8, 16, 5)  # output 64x32
        self.down3 = DownResBlock(16, 32, 5, new_resnet=new_resnet)  # output 32x16
        self.down4 = DownResBlock(32, 64, 5, new_resnet=new_resnet)  # output 16x8
        self.down5 = DownResBlock(64, 128, 5, new_resnet=new_resnet)  # output 8x4
        self.down6 = DownResBlock(128, 256, 5, new_resnet=new_resnet)  # output 4x2
        self.tail = ClassifyTailBlock(256, 2)  # output 1x1

    def forward(self, x):
        x = self.databn(x)
        x = self.conv1(x)
        x = self.pool2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        x = self.down6(x)
        x = self.tail(x)
        x = torch.flatten(x, 1)
        output = F.log_softmax(x, dim=1)
        return output