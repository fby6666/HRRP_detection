import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F
#多通道并行卷积模块
#用3，5，7卷积核进行多通道并行卷积，实现通道增加但是长度不变
#这里每一层卷积输出通道数暂设32
class multiScaleConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        """
        多通道并行卷积
        :param in_channel: 输入通道数量
        :param out_channel: 输出通道数量
        """
        super().__init__()
        self.branch1=nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channel),
            nn.ReLU()
        )
        self.branch2=nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(out_channel),
            nn.ReLU()
        )
        self.branch3=nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(out_channel),
            nn.ReLU()
        )
    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        return torch.cat((out1, out2, out3), 1)

#完整的网络
#多通道并行卷积后，再进行一次卷积，使其输出通道数减少，同时每个通道的特征维度也进行一定压缩  （进行压缩的卷积的参数如何选取效果较好？）
#展平维数，添加一个隐藏层，隐藏单元的数量这里暂给128  （隐藏单元选取数量多少效果较好？）
#输出节点只有一个
# class HRRP_CNN(nn.Module):
#     def __init__(self, input_length):
#         """
#         完整的HRRP检测网络
#         :param input_length: 输入一维HRRP信号的长度
#         """
#         super().__init__()
#         #多尺度并行卷积，中每个卷积的输出通道数量为out_channel
#         self.features=nn.Sequential(
#             multiScaleConv(in_channel=1, out_channel=32),
#             nn.MaxPool1d(2),
#             # nn.AvgPool1d(2),
#             #第二层卷积
#             nn.Conv1d(32*3,64,kernel_size=5,stride=1,padding=2),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Flatten(),
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(64*(input_length//4),128),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(128,1),
#             nn.Sigmoid()
#         )
#     def forward(self, x):
#         x=self.features(x)
#         return self.classifier(x)

class HRRP_CNN_allbands(nn.Module):
    def __init__(self, input_length):
        """
        完整的HRRP检测网络
        :param input_length: 输入一维HRRP信号的长度
        """
        super().__init__()
        #多尺度并行卷积，中每个卷积的输出通道数量为out_channel
        self.features=nn.Sequential(
            multiScaleConv(in_channel=4, out_channel=4),
            nn.MaxPool1d(2),
            # nn.AvgPool1d(2),
            #第二层卷积
            nn.Conv1d(4*3,6,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm1d(6),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(6*(input_length//4),32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32,1),
            # nn.Sigmoid()
        )
    def forward(self, x):
        x=self.features(x)
        return self.classifier(x)

class CNN_easy(nn.Module):
    def __init__(self, input_length):
        """
        完整的HRRP检测网络
        :param input_length: 输入一维HRRP信号的长度
        """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=3, stride=1, padding=1),nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(2),
            # nn.AvgPool1d(2),
            # 第二层卷积
            # nn.Conv1d(16, 6, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm1d(6),
            # nn.ReLU(),
            # nn.MaxPool1d(2),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 *(input_length // 2), 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x=self.features(x)
        return self.classifier(x)

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv1d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.BatchNorm1d(out_channels))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool1d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

class HRRP_VGG11(nn.Module):
    def __init__(self, input_length,conv_arch=((1, 16), (1, 32), (2, 64), (2, 128), (2, 128))):
        """
        完整的HRRP检测网络
        :param input_length: 输入一维HRRP信号的长度
        """
        super().__init__()
        conv_blks = []
        in_channels = 4
        for (num_convs, out_channels) in conv_arch:
            conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels
        self.features=nn.Sequential(*conv_blks,nn.Flatten())
        self.classifier=nn.Sequential(
            nn.Linear(128 * (input_length // 32), 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )
    def forward(self, x):
        x=self.features(x)
        return self.classifier(x)

# 残差块
class HRRP_Residual(nn.Module):  #@save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv1d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv1d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm1d(num_channels)
        self.bn2 = nn.BatchNorm1d(num_channels)
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(HRRP_Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(HRRP_Residual(num_channels, num_channels))
    return blk

class HRRP_ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        b1 = nn.Sequential(nn.Conv1d(4, 64, kernel_size=7, stride=2, padding=3),
                           nn.BatchNorm1d(64), nn.ReLU(),
                           nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        b3 = nn.Sequential(*resnet_block(64, 128, 2))
        b4 = nn.Sequential(*resnet_block(128, 256, 2))
        b5 = nn.Sequential(*resnet_block(256, 512, 2))
        self.features=nn.Sequential(b1, b2, b3, b4, b5,
                                    nn.AdaptiveAvgPool1d(1),
                                    nn.Flatten())
        self.classifier=nn.Sequential(
            nn.Linear(512, 1),
        )
    def forward(self, x):
        x=self.features(x)
        return self.classifier(x)