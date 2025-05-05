import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F
from model import HRRP_Residual,resnet_block,vgg_block
class CNN_easy(nn.Module):
    def __init__(self, input_length, use_se=False, feature_dim=32):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.Sequential(
            nn.BatchNorm1d(16),
            nn.MaxPool1d(2),
        )
        self.flatten = nn.Flatten()
        self.feature_extractor = nn.Sequential(
            nn.Linear(16 * (input_length // 2), feature_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.flatten(x)
        features = self.feature_extractor(x)
        return features

class HRRP_VGG11_backbone(nn.Module):
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
        # self.classifier=nn.Sequential(
        #     nn.Linear(128 * (input_length // 32), 128),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(128, 1),
        # )
    def forward(self, x):
        x=self.features(x)
        return x
class HRRP_ResNet_backbone(nn.Module):
    def __init__(self):
        super().__init__()
        b1 = nn.Sequential(nn.Conv1d(4, 64, kernel_size=7, stride=2, padding=3),
                           nn.BatchNorm1d(64), nn.ReLU(),
                           nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        b2 = nn.Sequential(*resnet_block(64, 64, 2, use_se=False, first_block=True))
        b3 = nn.Sequential(*resnet_block(64, 128, 2, use_se=False))
        b4 = nn.Sequential(*resnet_block(128, 256, 2, use_se=False))
        b5 = nn.Sequential(*resnet_block(256, 512, 2, use_se=False))

        self.features = nn.Sequential(
            b1, b2, b3, b4, b5,
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

    def forward(self, x):
        x = self.features(x)
        return x  # 注意这里只返回特征向量！
class ProjectionHead(nn.Module):
    def __init__(self, feature_dim=128, proj_dim=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x):
        x = self.proj(x)
        return nn.functional.normalize(x, dim=1)  # L2归一化

class SupConModel(nn.Module):
    def __init__(self, input_length, feature_dim, proj_dim):
        super(SupConModel, self).__init__()
        self.backbone = HRRP_ResNet_backbone()
        self.projection_head = ProjectionHead(feature_dim, proj_dim)

    def forward(self, x):
        features = self.backbone(x)  # [batch_size, feature_dim]
        projections = self.projection_head(features)  # [batch_size, proj_dim]
        return projections