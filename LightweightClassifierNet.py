import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBn2d(nn.Sequential):
    """
    卷积层+批归一化层的组合
    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小，默认为(1,1)
        stride: 步长，默认为(1,1)
        padding: 填充大小，默认为(0,0)
    """

    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)):
        super(ConvBn2d, self).__init__(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                                       nn.BatchNorm2d(out_channels))


class ConvBnReLU2d(nn.Sequential):
    """
    卷积层+批归一化层+ReLU激活层的组合
    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小，默认为(1,1)
        stride: 步长，默认为(1,1)
        padding: 填充大小，默认为(0,0)
    """

    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)):
        super(ConvBnReLU2d, self).__init__(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                                           nn.BatchNorm2d(out_channels), nn.ReLU())


class PoolResBlock(nn.Module):
    """
    池化残差块
    包含最大池化层和三个分支的残差连接
    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super(PoolResBlock, self).__init__()
        # 最大池化层，kernel_size=3, stride=2
        self.pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), ceil_mode=True)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.1)  # Dropout层，防止过拟合
        
        # 三个分支
        # 分支0: 1x1卷积
        self.branch0 = ConvBn2d(in_channels, out_channels, kernel_size=(1, 1))
        # 分支1: 先水平后垂直的卷积
        self.branch1 = nn.Sequential(
            ConvBnReLU2d(in_channels, out_channels // 4, kernel_size=(kernel_size, 1), padding=(kernel_size // 2, 0)),
            ConvBnReLU2d(out_channels // 4, out_channels // 4, kernel_size=(1, kernel_size),
                         padding=(0, kernel_size // 2)))
        # 分支2: 先垂直后水平的卷积
        self.branch2 = nn.Sequential(
            ConvBnReLU2d(in_channels, out_channels // 4, kernel_size=(1, kernel_size), padding=(0, kernel_size // 2)),
            ConvBnReLU2d(out_channels // 4, out_channels // 4, kernel_size=(kernel_size, 1),
                         padding=(kernel_size // 2, 0)))
        # 合并分支1和分支2的输出
        self.merge12 = ConvBn2d(out_channels // 4, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        # 前向传播
        out = self.pool(x)  # 最大池化
        branch0 = self.branch0(out)  # 分支0
        branch1 = self.branch1(out)  # 分支1
        branch2 = self.branch2(out)  # 分支2
        # 合并所有分支
        out = branch0 + self.merge12(branch1 + branch2)
        out = self.relu(out)  # ReLU激活
        out = self.drop(out)  # Dropout
        return out


class DownResBlock(nn.Module):
    """
    下采样残差块
    用于降低特征图尺寸并增加通道数
    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        new_resnet: 是否使用新的ResNet结构
    """

    def __init__(self, in_channels, out_channels, kernel_size, new_resnet=True):
        super(DownResBlock, self).__init__()
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.1)
        
        # 分支0: 下采样路径
        if new_resnet:
            # 新的ResNet结构使用平均池化
            self.branch0 = nn.Sequential(nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
                                         ConvBn2d(in_channels, out_channels, kernel_size=(1, 1)))
        else:
            # 传统结构使用stride=2的1x1卷积
            self.branch0 = ConvBn2d(in_channels, out_channels, kernel_size=(1, 1), stride=(2, 2))
            
        # 分支1和分支2: 使用分离卷积进行下采样
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
        # 合并分支1和分支2的输出
        self.merge12 = ConvBn2d(out_channels // 4, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        # 前向传播
        branch0 = self.branch0(x)  # 下采样分支
        branch1 = self.branch1(x)  # 分支1
        branch2 = self.branch2(x)  # 分支2
        # 合并所有分支
        out = branch0 + self.merge12(branch1 + branch2)
        out = self.relu(out)  # ReLU激活
        out = self.drop(out)  # Dropout
        return out


class ClassifyTailBlock(nn.Module):
    """
    分类尾部块
    用于最终的特征提取和分类
    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数（类别数）
    """

    def __init__(self, in_channels, out_channels):
        super(ClassifyTailBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))  # 1x1卷积
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化

    def forward(self, x):
        x = self.conv(x)  # 1x1卷积
        x = self.relu(x)  # ReLU激活
        x = self.pool(x)  # 全局平均池化
        return x


class LargeKernelClassifierNet(nn.Module):
    """
    大卷积核分类网络
    整个网络的主体架构
    参数:
        new_resnet: 是否使用新的ResNet结构
    """

    def __init__(self, new_resnet=False):
        super(LargeKernelClassifierNet, self).__init__()
        # 数据批归一化
        self.databn = nn.BatchNorm2d(3)  # 输出 256x128
        
        # 初始卷积层
        if new_resnet:
            # 新的ResNet结构使用多个3x3卷积
            self.conv1 = nn.Sequential(
                ConvBnReLU2d(3, 4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                ConvBnReLU2d(4, 4, kernel_size=(3, 3), padding=(1, 1)),
                ConvBnReLU2d(4, 8, kernel_size=(3, 3), padding=(1, 1)),
                nn.Dropout(p=0.1))  # 输出 128x64
        else:
            # 传统结构使用单个7x7卷积
            self.conv1 = nn.Sequential(
                ConvBnReLU2d(3, 8, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)),
                nn.Dropout(p=0.1))  # 输出 128x64
            
        # 网络主体结构
        self.pool2 = PoolResBlock(8, 16, 5)    # 输出 64x32
        self.down3 = DownResBlock(16, 32, 5, new_resnet=new_resnet)  # 输出 32x16
        self.down4 = DownResBlock(32, 64, 5, new_resnet=new_resnet)  # 输出 16x8
        self.down5 = DownResBlock(64, 128, 5, new_resnet=new_resnet)  # 输出 8x4
        self.down6 = DownResBlock(128, 256, 5, new_resnet=new_resnet)  # 输出 4x2
        self.tail = ClassifyTailBlock(256, 2)  # 输出 1x1

    def forward(self, x):
        # 前向传播
        x = self.databn(x)  # 数据批归一化
        x = self.conv1(x)   # 初始卷积
        x = self.pool2(x)   # 池化残差块
        x = self.down3(x)   # 下采样残差块3
        x = self.down4(x)   # 下采样残差块4
        x = self.down5(x)   # 下采样残差块5
        x = self.down6(x)   # 下采样残差块6
        x = self.tail(x)    # 分类尾部
        x = torch.flatten(x, 1)  # 展平
        output = F.log_softmax(x, dim=1)  # 对数softmax
        return output