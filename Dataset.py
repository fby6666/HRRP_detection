import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
def load_h5_data(file_path, dataset_name):
    """ 读取 HDF5 文件中的数据，并转换为 (n, 4, 210) 形式 """
    with h5py.File(file_path, 'r') as f:
        data = np.array(f[dataset_name])  # 读取数据
        print(f"原始数据 {dataset_name} 形状: {data.shape}")

        # 需要转换数据形状
        if dataset_name == "/tg_include":
            data = np.transpose(data, (3, 2, 0, 1))  # (31, 3600, 4, 210)
        else:
            data = np.transpose(data, (2, 0, 1))  # (n, 4, 210)

        print(f"转换后 {dataset_name} 形状: {data.shape}")
        return data
class RadarDataset_allbands(Dataset):
    def __init__(self, features, labels):
        """
        初始化数据集
        :param features: 形状 (num_samples, 4, 210) 的特征数据
        :param labels: 形状 (num_samples,) 的标签数据
        """
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.float32)

    def __len__(self):
        """返回数据集大小"""
        return len(self.features)

    def __getitem__(self, idx):
        """获取索引对应的样本和标签"""
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y
def get_dataloaders(train_h5, test_h5, batch_size=32, shuffle=True):
    """
    加载数据集，并划分为训练集和不同 SNR 条件的测试集
    :param train_h5: 训练数据 HDF5 文件路径
    :param test_h5: 测试数据 HDF5 文件路径
    :param batch_size: batch size
    :param shuffle: 是否打乱数据
    :return: 训练 DataLoader, {信杂比: 测试 DataLoader}
    """
    # 读取训练数据
    train_clutter = load_h5_data(train_h5, "/train_cl")  # (18000, 4, 210)
    train_target = load_h5_data(train_h5, "/train_tg")  # (720, 4, 210)
    X_train = np.concatenate([train_clutter, train_target], axis=0)
    y_train = np.concatenate([np.zeros(train_clutter.shape[0]), np.ones(train_target.shape[0])], axis=0)
    print(f"最终训练数据形状: {X_train.shape}")

    # 创建训练 DataLoader
    train_dataset = RadarDataset_allbands(X_train, y_train)
    # print('tarin,label ',train_dataset.__getitem__(1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    # 读取测试数据
    test_clutter = load_h5_data(test_h5, "/clutter")  # (2000, 4, 210)
    tg_data_processed = load_h5_data(test_h5, "/tg_include")  # (31, 3600, 4, 210)

    # 处理不同信杂比的测试数据
    # SCR_list = -10:20; %dB
    test_loaders = {}
    for i in range(tg_data_processed.shape[0]):  # 遍历 31 个信杂比条件
        X_test = np.concatenate([test_clutter, tg_data_processed[i]], axis=0)  # (5600, 4, 210)
        y_test = np.concatenate([np.zeros(test_clutter.shape[0]), np.ones(tg_data_processed[i].shape[0])], axis=0)

        test_dataset = RadarDataset_allbands(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        test_loaders[f"SNR_{i}"] = test_loader
        # print(f"信杂比 {i} 测试数据形状: {X_test.shape}")

    return train_loader, test_loaders

# train_loader, test_loaders = get_dataloaders("train_data.h5", "test_data.h5",batch_size=64)
# 遍历不同信杂比的测试数据
# print(test_loaders)
