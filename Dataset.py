import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
def load_h5_test_data(file_path, dataset_name):
    """ 读取 HDF5 文件中的数据，并转换为 (n, 4, 210) 形式 """
    with h5py.File(file_path, 'r') as f:
        data = np.array(f[dataset_name])  # 读取数据
        print(f"原始数据 {dataset_name} 形状: {data.shape}")

        # 需要转换数据形状
        if dataset_name != "/cl":
            data = np.transpose(data, (2, 1, 3, 0))  # (31, 3600, 4, 210)
        else:
            data = np.transpose(data, (2,1,0))  # (n, 4, 210)

        print(f"转换后 {dataset_name} 形状: {data.shape}")
        return data

def load_h5_data(file_path, dataset_name):
    """ 读取 HDF5 文件中的数据，并转换为 (n, 4, 210) 形式 """
    with h5py.File(file_path, 'r') as f:
        data = np.array(f[dataset_name])  # 读取数据
        print(f"原始数据 {dataset_name} 形状: {data.shape}")
        # 需要转换数据形状
        if dataset_name == '/cl':
            data=np.transpose(data,(2,1,0))
        else:
            data=np.transpose(data,(1,2,0))
        print(f"转换后 {dataset_name} 形状: {data.shape}")
        return data
class RadarDataset_allbands(Dataset):
    def __init__(self, features, labels):
        self.features = features  # 不再转为 float32
        self.labels = labels
        assert len(self.features) == len(self.labels), "数据和标签长度不一致！"
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # 在这里转为 float32 tensor，避免一次性内存压力
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y
# class RadarDataset_allbands(Dataset):
#     def __init__(self, features, labels):
#         """
#         初始化数据集
#         :param features: 形状 (num_samples, 4, 210) 的特征数据
#         :param labels: 形状 (num_samples,) 的标签数据
#         """
#         self.features = features.astype(np.float32)
#         self.labels = labels.astype(np.float32)
#
#     def __len__(self):
#         """返回数据集大小"""
#         return len(self.features)
#
#     def __getitem__(self, idx):
#         """获取索引对应的样本和标签"""
#         x = torch.tensor(self.features[idx], dtype=torch.float32)
#         y = torch.tensor(self.labels[idx], dtype=torch.float32)
#         return x, y
def get_dataloaders_test(test_h5, batch_size=32, shuffle=True):
    """
    加载数据集，并划分为训练集和不同 SNR 条件的测试集
    :param test_h5: 测试数据 HDF5 文件路径
    :param batch_size: batch size
    :param shuffle: 是否打乱数据
    :return: 训练 DataLoader, {信杂比: 测试 DataLoader}
    """
    # 读取训练数据
    clutter = load_h5_test_data(test_h5, "/cl")  # (18000, 4, 210)
    target_ZB = load_h5_test_data(test_h5, "/tg_cl_zb")
    target_ZL = load_h5_test_data(test_h5, "/tg_cl_zl")
    target_FB = load_h5_test_data(test_h5, "/tg_cl_fb")
    target_FL = load_h5_test_data(test_h5, "/tg_cl_fl")
    y_cl=np.zeros(clutter.shape[0])
    # y_train = np.concatenate([np.zeros(train_clutter.shape[0]), np.ones(train_target.shape[0])], axis=0)
    # 创建训练 DataLoader
    clutter_dataset = RadarDataset_allbands(clutter, y_cl)
    clutter_loader = DataLoader(clutter_dataset, batch_size=batch_size, shuffle=shuffle)

    # 读取叠加信号数据
    tg =np.concatenate([target_ZB,target_ZL,target_FB,target_FL],axis=1)
    y_tg=np.ones(tg.shape[1])
    print(f"最终叠加信号数据形状: {tg.shape}")
    # tg_data_processed = load_h5_data(test_h5, "/tg_include")  # (31, 3600, 4, 210)

    # 处理不同信杂比的测试数据
    # SCR_list = -10:20; %dB
    test_loaders = {}
    for i in range(tg.shape[0]):  # 遍历 31 个信杂比条件
        test_tg_dataset = RadarDataset_allbands(tg[i], y_tg)
        test_loader = DataLoader(test_tg_dataset, batch_size=batch_size, shuffle=False)
        test_loaders[f"SNR_{i}"] = test_loader
        # print(f"信杂比 {i} 测试数据形状: {test_loader.__len__()}")

    return clutter_loader, test_loaders
# def get_dataloaders(train_h5, val_h5, batch_size=32, shuffle=True):
#     """
#     加载数据集，并划分为训练集和不同 SNR 条件的测试集
#     :param train_h5: 训练数据 HDF5 文件路径
#     :param test_h5: 测试数据 HDF5 文件路径
#     :param batch_size: batch size
#     :param shuffle: 是否打乱数据
#     :return: 训练 DataLoader, {信杂比: 测试 DataLoader}
#     """
#     # 读取训练数据
#     train_clutter = load_h5_data(train_h5, "/cl")  # (18000, 4, 210)
#     train_target_ZB = load_h5_data(train_h5, "/tg_cl_zb")
#     train_target_ZL = load_h5_data(train_h5, "/tg_cl_zl")
#     train_target_FB = load_h5_data(train_h5, "/tg_cl_fb")
#     train_target_FL = load_h5_data(train_h5, "/tg_cl_fl")
#     X_train = np.concatenate([train_clutter, train_target_ZB,train_target_ZL,train_target_FB,train_target_FL], axis=0)
#     y_train = np.concatenate([np.zeros(train_clutter.shape[0]), np.ones(4*train_target_FL.shape[0])], axis=0)
#     print(f"最终训练集数据形状: {X_train.shape}")
#
#     # 创建训练 DataLoader
#     train_dataset = RadarDataset_allbands(X_train, y_train)
#     # print('tarin,label ',train_dataset.__getitem__(1))
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
#
#     # 读取测试数据
#     val_clutter = load_h5_data(val_h5, "/cl")  # (2000, 4, 210)
#     val_target_ZB = load_h5_data(val_h5, "/tg_cl_zb")
#     val_target_ZL = load_h5_data(val_h5, "/tg_cl_zl")
#     val_target_FB = load_h5_data(val_h5, "/tg_cl_fb")
#     val_target_FL = load_h5_data(val_h5, "/tg_cl_fl")
#     X_val = np.concatenate([val_clutter, val_target_ZB,val_target_ZL,val_target_FB,val_target_FL], axis=0)
#     y_val = np.concatenate([np.zeros(val_clutter.shape[0]), np.ones(4*val_target_ZB.shape[0])], axis=0)
#     print(f"最终验证集数据形状: {X_val.shape}")
#     # 创建验证 DataLoader
#     val_dataset = RadarDataset_allbands(X_val, y_val)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
#
#     return train_loader, val_loader

def get_dataloaders(train_h5, val_h5, batch_size=32, shuffle=True):
    """
    加载数据集，并划分为训练集和不同 SNR 条件的测试集
    :param train_h5: 训练数据 HDF5 文件路径
    :param val_h5: 测试数据 HDF5 文件路径
    :param batch_size: batch size
    :param shuffle: 是否打乱数据
    :return: 训练 DataLoader 字典（按SNR分割）, 验证 DataLoader
    """
    # 读取训练数据
    train_clutter = load_h5_data(train_h5, "/cl")  # (18000, 4, 210)
    train_target_ZB = load_h5_data(train_h5, "/tg_cl_zb")
    train_target_ZL = load_h5_data(train_h5, "/tg_cl_zl")
    train_target_FB = load_h5_data(train_h5, "/tg_cl_fb")
    train_target_FL = load_h5_data(train_h5, "/tg_cl_fl")

    # 按SNR分割训练数据
    # snr_ranges = [(-10, 0), (0,10), (10,20)]  # 定义SNR范围
    # base+se
    snr_ranges = [(-10, 20)]  # 定义SNR范围
    train_loaders = {}
    snr_values = np.linspace(-10, 20, 31)
    samples_per_snr = 900  # 每个 SNR 级别有 900 个样本
    for snr_min, snr_max in snr_ranges:
        # 选择对应SNR范围的样本索引
        snr_indices = []
        for i, snr in enumerate(snr_values):
            if snr_min <= snr <= snr_max:
                # 每个 SNR 级别的样本索引范围：i * 900 到 (i + 1) * 900 - 1
                start_idx = i * samples_per_snr
                end_idx = (i + 1) * samples_per_snr
                snr_indices.extend(range(start_idx, end_idx))
        print(snr_indices)
        if not snr_indices:
            print(f"警告: SNR 范围 [{snr_min}, {snr_max}] 没有匹配的数据")
            continue
        print(f"SNR 范围 [{snr_min}, {snr_max}] 的样本索引数量: {len(snr_indices)}")

        # 提取对应SNR的样本
        target_ZB_snr = train_target_ZB[snr_indices]  # 形状: (num_samples, 4, 210)
        target_ZL_snr = train_target_ZL[snr_indices]
        target_FB_snr = train_target_FB[snr_indices]
        target_FL_snr = train_target_FL[snr_indices]

        # 合并杂波和目标数据
        X_train_snr = np.concatenate([train_clutter, target_ZB_snr, target_ZL_snr, target_FB_snr, target_FL_snr],
                                     axis=0)
        y_train_snr = np.concatenate([np.zeros(train_clutter.shape[0]), np.ones(4 * target_ZB_snr.shape[0])], axis=0)

        print(f"SNR 范围 [{snr_min}, {snr_max}] 数据形状: {X_train_snr.shape}")

        # 创建数据集和DataLoader
        train_dataset_snr = RadarDataset_allbands(X_train_snr, y_train_snr)
        train_loader_snr = DataLoader(train_dataset_snr, batch_size=batch_size, shuffle=shuffle)
        train_loaders[f"SNR_{snr_min}_{snr_max}"] = train_loader_snr

    # 读取验证数据（保持不变）
    val_clutter = load_h5_data(val_h5, "/cl")  # (2000, 4, 210)
    val_target_ZB = load_h5_data(val_h5, "/tg_cl_zb")
    val_target_ZL = load_h5_data(val_h5, "/tg_cl_zl")
    val_target_FB = load_h5_data(val_h5, "/tg_cl_fb")
    val_target_FL = load_h5_data(val_h5, "/tg_cl_fl")
    X_val = np.concatenate([val_clutter, val_target_ZB, val_target_ZL, val_target_FB, val_target_FL], axis=0)
    y_val = np.concatenate([np.zeros(val_clutter.shape[0]), np.ones(4 * val_target_ZB.shape[0])], axis=0)
    print(f"最终验证集数据形状: {X_val.shape}")

    # 创建验证 DataLoader
    val_dataset = RadarDataset_allbands(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loaders, val_loader

# train_loader, test_loaders = get_dataloaders("train_data.h5", "test_data.h5",batch_size=64)
# 遍历不同信杂比的测试数据
# print(test_loaders)
if __name__ == "__main__":
    get_dataloaders("train.h5","val.h5",batch_size=64)
