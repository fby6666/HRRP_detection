import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from  Dataset import *
# from train import *
from d2l import torch as d2l
from model_construst import *
from model import *
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
titles = [
    "VGG11 (No SupCon)", "VGG11 (With SupCon)",
    "ResNet18 (No SupCon)", "ResNet18 (With SupCon)"
]
img_paths = [
    "tsne_visualization_VGG11.png", "tsne_visualization_VGG11_SCL.png",
    "tsne_visualization_ResNet18.png", "tsne_visualization_ResNet18_SCL.png"
]

for ax, path, title in zip(axs.flat, img_paths, titles):
    img = mpimg.imread(path)
    ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.savefig("tsne_merged.png", dpi=300)
plt.show()
# def get_dataloader_tsne(test_h5="test.h5",batch_size=32,shuffle=True):
#     clutter = load_h5_test_data(test_h5, "/cl") [0:800] # (18000, 4, 210)
#     target_ZB = load_h5_test_data(test_h5, "/tg_cl_zb")[0][0:200]
#     target_ZL = load_h5_test_data(test_h5, "/tg_cl_zl")[0][0:200]
#     target_FB = load_h5_test_data(test_h5, "/tg_cl_fb")[0][0:200]
#     target_FL = load_h5_test_data(test_h5, "/tg_cl_fl")[0][0:200]
#     print(f"叠加信号数据形状: {target_ZB.shape}")
#     print(f"杂波数据形状: {clutter.shape}")
#     X=np.concatenate([clutter,target_ZB,target_ZL,target_FB,target_FL],axis=0)
#     print(f"叠加信号数据形状: {X.shape}")
#     y=np.concatenate([np.zeros(clutter.shape[0]),np.ones(4*target_FL.shape[0])],axis=0)
#     print(f"叠加信号数据形状: {y.shape}")
#     dataset=RadarDataset_allbands(X,y)
#     data_loader=DataLoader(dataset,batch_size=batch_size,shuffle=shuffle)
#     return data_loader
# data_loader=get_dataloader_tsne()
# device = d2l.try_gpu()
# # weight_path= 'SCL/backbone_resnet18.pth'
# # weight_path= 'SCL/backbone_VGG11.pth'
# # weight_path='saved_models/best_VGG11_snr_-10_20.pth'
# weight_path='saved_models/best_Resnet18.pth'
# # model=init_model(input_length=210,model_num=2)
# # model=HRRP_VGG11(input_length=210)
# # model=HRRP_ResNet_backbone()
# # model=HRRP_VGG11(input_length=210)
# model=HRRP_ResNet()
# model.to(device)
# model.load_state_dict(torch.load(weight_path, weights_only=True))
# model.eval()
#
# # 1. 创建一个钩子来获取倒数第二层的特征输出
# features = []
# labels=[]
# # 1. 创建 hook 函数，注册到 self.pool 层
# def hook_fn(module, input, output):
#     # output.shape: (batch_size, 16, length)
#     # 做 flatten，使每个样本变成一维向量
#     pooled = output.detach().cpu()
#     pooled = pooled.view(pooled.size(0), -1)  # 展平为 (batch_size, features)
#     features.append(pooled)
# # 注册 hook 到 model.pool
# hook_handle = model.features.register_forward_hook(hook_fn)
# # hook_handle = model.features.register_forward_hook(hook_fn)
# # 2. 遍历数据提取特征
# with torch.no_grad():
#     for batch_x, batch_y in data_loader:
#         batch_x = batch_x.to(device)
#         batch_y = batch_y.to(device)
#         _ = model(batch_x)  # 自动触发 hook
#         labels.extend(batch_y.cpu().numpy())
#
# # 拼接所有特征
# features = torch.cat(features, dim=0).numpy()
# labels = np.array(labels)
#
# # 3. t-SNE 可视化
# tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
# features_2d = tsne.fit_transform(features)
#
# # 4. 绘图
# plt.figure(figsize=(8, 6))
# scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
# plt.legend(*scatter.legend_elements(), title="Class")
# plt.title("t-SNE Visualization of  Features (from pool)")
# plt.xlabel("Dim 1")
# plt.ylabel("Dim 2")
# plt.grid(True)
# plt.savefig('tsne_visualization_ResNet18.png')
# plt.show()
#
# # 移除 hook
# hook_handle.remove()