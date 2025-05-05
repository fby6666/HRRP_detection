import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F

class SEBlock1D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock1D, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.dropout=nn.Dropout(0.3)

    def forward(self, x):  # x: [B, C, L]
        b, c, l = x.size()
        y = self.global_pool(x).view(b, c)         # [B, C]
        y = self.fc1(y)
        y = self.relu(y)
        y=self.dropout(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1)          # [B, C, 1]
        return x * y.expand_as(x)                  # 通道加权

class DualSEBlock1D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(DualSEBlock1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.shared_mlp = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),nn.Dropout(0.3),
            nn.Linear(channel // reduction, channel, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x: [B, C, L]
        b, c, l = x.size()
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)

        # 注意力融合
        attn_avg = self.shared_mlp(avg_out)
        attn_max = self.shared_mlp(max_out)

        attn = self.sigmoid(attn_avg + attn_max).view(b, c, 1)
        return x * attn.expand_as(x)

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

# class CNN_easy(nn.Module):
#     def __init__(self, input_length):
#         """
#         完整的HRRP检测网络
#         :param input_length: 输入一维HRRP信号的长度
#         """
#         super().__init__()
#         self.features = nn.Sequential(
#             nn.Conv1d(4, 16, kernel_size=3, stride=1, padding=1),nn.ReLU(),
#             nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(),
#             nn.BatchNorm1d(16),
#             nn.MaxPool1d(2),
#             # nn.AvgPool1d(2),
#             # 第二层卷积
#             # nn.Conv1d(16, 6, kernel_size=5, stride=1, padding=2),
#             # nn.BatchNorm1d(6),
#             # nn.ReLU(),
#             # nn.MaxPool1d(2),
#             nn.Flatten(),
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(16 *(input_length // 2), 32),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(32, 1),
#             # nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         x=self.features(x)
#         return self.classifier(x)

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
# class CNN_easy(nn.Module):
#     def __init__(self, input_length, use_se=False):
#         super().__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv1d(4, 16, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#         )
#
#         self.conv2 = nn.Sequential(
#             nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#         )
#
#         self.se = SEBlock1D(16,reduction=4) if use_se else None
#         # self.se = DualSEBlock1D(16,reduction=4) if use_se else None
#         self.pool = nn.Sequential(
#             nn.BatchNorm1d(16),
#             nn.MaxPool1d(2),
#         )
#
#         self.flatten = nn.Flatten()
#         self.classifier = nn.Sequential(
#             nn.Linear(16 * (input_length // 2), 32),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(32, 1),
#         )
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         if self.se is not None:
#             x = self.se(x)
#         # x = self.se(x)
#         x = self.pool(x)
#         x = self.flatten(x)
#         return self.classifier(x)
def vgg_block(num_convs, in_channels, out_channels,use_se=False):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv1d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.BatchNorm1d(out_channels))
        layers.append(nn.ReLU())
        in_channels = out_channels
    if use_se:
        layers.append(DualSEBlock1D(out_channels))
        # layers.append(SEBlock1D(out_channels))
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
            # 决定是否添加SE注意力机制
            conv_blks.append(vgg_block(num_convs, in_channels, out_channels,use_se=False))
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
                 use_1x1conv=False,use_se=False ,strides=1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv1d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_se:
            # self.SE=SEBlock1D(num_channels)
            self.SE = DualSEBlock1D(num_channels)
        else :
            self.SE=None
        if use_1x1conv:
            self.conv3 = nn.Conv1d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
            # self.bn3 = nn.BatchNorm1d(num_channels)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm1d(num_channels)
        self.bn2 = nn.BatchNorm1d(num_channels)
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.SE != None:
            Y=self.SE(Y)
        if self.conv3:
            # X = self.bn3(self.conv3(X))
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
# 决定是否添加SE注意力机制
def resnet_block(input_channels, num_channels, num_residuals,use_se=False,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(HRRP_Residual(input_channels, num_channels,
                                use_1x1conv=True,use_se=use_se,strides=2))
        else:
            blk.append(HRRP_Residual(num_channels, num_channels,use_se=use_se))
    return blk

class HRRP_ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        b1 = nn.Sequential(nn.Conv1d(4, 64, kernel_size=7, stride=2, padding=3),
                           nn.BatchNorm1d(64), nn.ReLU(),
                           nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        b2 = nn.Sequential(*resnet_block(64, 64, 2,use_se=False,first_block=True))
        b3 = nn.Sequential(*resnet_block(64, 128, 2,use_se=False))
        b4 = nn.Sequential(*resnet_block(128, 256, 2,use_se=False))
        b5 = nn.Sequential(*resnet_block(256, 512, 2,use_se=False))
        self.features=nn.Sequential(b1, b2, b3, b4, b5,
                                    nn.AdaptiveAvgPool1d(1),
                                    nn.Flatten())
        self.classifier=nn.Sequential(
            nn.Linear(512, 1),
        )
    def forward(self, x):
        x=self.features(x)
        return self.classifier(x)