import numpy as np
from model import *
from d2l import torch as d2l
import torch
from matplotlib import pyplot as plt
import os
import time
import random

#初始化模型
def init_model(input_length,model_num=4):
    """
    初始化模型
    :return: 返回初始化好参数的模型 ，并未传入GPU
    """
    if model_num==1:
        net= HRRP_ResNet()
    elif model_num==2:
        net =CNN_easy(input_length=input_length)
    elif model_num==3:
        net = HRRP_VGG11(input_length=input_length)
    elif model_num==4:
        net=HRRP_CNN_allbands(input_length=input_length)
    #初始化模型参数
    def init_weight(m):
        if isinstance (m, nn.Conv1d) or isinstance (m,nn.Linear):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance (m,nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    net.apply(init_weight)
    return net

def save_model(net, save_dir='./saved_models', model_name=None):
    """
    保存模型参数到指定目录
    参数:
        net: 要保存的模型实例
        save_dir: 保存目录（默认为当前目录下的saved_models）
        model_name: 模型文件名（若为None，则按时间戳生成）
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # 自动创建目录

    if model_name is None:
        # 生成唯一文件名（时间戳格式）
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_name = f"hrrp_model_{timestamp}.pth"

    save_path = os.path.join(save_dir, model_name)
    torch.save(net.state_dict(), save_path)  # 仅保存模型参数
    print(f"模型已保存至: {save_path}")


class CustomLoss(nn.Module):
    """
    定义的损失函数
    二元交叉损失熵+lambda**(Pf-1e-4)
    """
    def __init__(self, lambda_fpr=0.1):
        super().__init__()
        self.lambda_fpr = lambda_fpr
        self.bce = nn.BCELoss()

    def forward(self, outputs, labels):
        # 计算交叉熵损失
        bce_loss = self.bce(outputs.squeeze(), labels)

        # 计算虚警率（FPR）
        pred_labels = (outputs >= 0.5).float()
        fp = ((pred_labels == 1) & (labels == 0)).float().sum()
        tn = labels.shape[0].float()
        fpr = fp / (tn + fp + 1e-7)  # 防止除零

        return bce_loss + self.lambda_fpr * (fpr-1e-4)**2

def train_no_sigmoid(net,model_name,train_iter,test_iter,
                     num_epochs,lr,device=d2l.try_gpu(),threshold=0.5):
    best_acc = 0.0
    print('training on ',device)
    net.to(device)
    optimizer=torch.optim.SGD(net.parameters(),lr=lr)
    pos_weight = torch.tensor([25.0]).to(device)  # 视数据不均衡程度调整
    loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # loss=nn.BCELoss()
    animator=d2l.Animator(xlabel='epoch',xlim=[1,num_epochs],ylim=[0,1],
    legend=['train_loss','train_acc','test_acc'])
    timer,num_batches=d2l.Timer(),len(train_iter)
    for epoch in range(num_epochs):
        metric=d2l.Accumulator(3)
        net.train()
        for i ,(X,y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X,y=X.to(device),y.to(device)
            y_hat=net(X).squeeze(dim=1)
            l=loss(y_hat,y)
            l.backward()
            optimizer.step()
            # 梯度裁剪（防止爆炸）
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            with torch.no_grad():
                # 计算训练精度
                y_pred = (torch.sigmoid(y_hat) >= threshold).float()  # 转换为 0/1
                metric.add(l*X.shape[0],d2l.accuracy(y_pred,y),X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc=evaluate_accuracy_gpu(net,test_iter,threshold,device)
        animator.add(epoch + 1, (None, None, test_acc))
        if test_acc>best_acc:
            best_acc=test_acc
            save_model(net,model_name=model_name)
            print(f"最佳测试准确率更新至: {best_acc:.3f}, 模型已保存")
        print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
        print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
              f'on {str(device)}')
    plt.savefig('accuracy.png')
    plt.show()
def save_pd_pfa_to_txt(filename, results):
    """将 pd 和 pfa 保存到 TXT 文件"""
    with open(filename, "w") as f:
        for model_name, (pd_list, pfa_list) in results.items():
            f.write(f"{model_name}\n")
            f.write("pd: " + ",".join(map(str, pd_list)) + "\n")
            f.write("pfa: " + ",".join(map(str, pfa_list)) + "\n\n")
def predict_scr_pd_pfa(weight_path,model_num,test_loaders,pic_name,
                       device=d2l.try_gpu(),input_length=210,threshold=0.999):
    models = {
        "4": "cnn_mul",
        "2": "cnn_easy",
        "3": "VGG11",
        "1": "Resnet18",
    }
    net = init_model(input_length, model_num=model_num)
    net.load_state_dict(torch.load(weight_path, weights_only=True))
    net.to(device)
    net.eval()
    print("模型权重已成功加载！")
    # 2. 存储测试精度
    scr_levels = sorted(test_loaders.keys(), key=lambda x: int(x.split('_')[1]))  # 按数字顺序排序 SNR
    # acc_list = []
    pd_list, pfa_list = [], []  # 用于存储不同 SNR 下的 Pd 和 Pfa
    snr_db = np.linspace(-10, 20, len(scr_levels))  # 将 SNR 映射到 -10 dB 到 20 dB
    # 3. 遍历不同 SNR 测试集
    with torch.no_grad():
        for snr in scr_levels:
            test_loader = test_loaders[snr]
            tp, fn, fp, tn = 0, 0, 0, 0  # 统计 TP, FN, FP, TN
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                y_hat = net(X).squeeze(dim=1)
                y_pred = (torch.sigmoid(y_hat) >= threshold).float()  # 计算预测值

                tp += ((y_pred == 1) & (y == 1)).sum().item()  # 真阳性 (TP)
                fn += ((y_pred == 0) & (y == 1)).sum().item()  # 漏检 (FN)
                fp += ((y_pred == 1) & (y == 0)).sum().item()  # 误报 (FP)
                tn += ((y_pred == 0) & (y == 0)).sum().item()  # 真阴性 (TN)

            # 计算 Pd (检测概率) 和 Pfa (虚警率)
            pd = tp / (tp + fn) if (tp + fn) > 0 else 0  # Pd = TP / (TP + FN)
            pfa = fp / (fp + tn) if (fp + tn) > 0 else 0  # Pfa = FP / (FP + TN)

            pd_list.append(pd)
            pfa_list.append(pfa)

            print(f"SNR {snr}: Pd = {pd:.3f}, Pfa = {pfa:.3f}")

    # 绘制 Pd 和 Pfa 随 SNR 变化的曲线
    plt.figure(figsize=(8, 6))
    plt.plot(snr_db, pd_list, marker='o', linestyle='-', color='b', label='Pd (Detection Probability)')
    plt.plot(snr_db, pfa_list, marker='s', linestyle='--', color='r', label='Pfa (False Alarm Rate)')

    plt.xlabel('SNR (dB)')
    plt.ylabel('Probability')
    plt.title('Detection Probability (Pd) and False Alarm Rate (Pfa) vs SNR')
    plt.legend()
    plt.grid(True)
    plt.savefig(pic_name)  # 保存图片
    plt.show()

    # 将pd_list,pfa_list写入txt文件
    with open("results_pd_pfa.txt", "a") as f:
        f.write(f"{models[str(model_num)]}:\n")
        f.write("pd: " + ",".join(map(str, pd_list)) + "\n")
        f.write("pfa: " + ",".join(map(str, pfa_list)) + "\n\n")


def predict_scr_clutter(weight_path,input_length,device=d2l.try_gpu(),threshold=0.999,model_num=4):
    net = init_model(input_length, model_num=model_num)
    net.load_state_dict(torch.load(weight_path, weights_only=True))
    net.to(device)
    net.eval()
    print("模型权重已成功加载！")
    test_clutter = load_h5_data("test_data.h5", "/clutter")  # (2000, 4, 210)
    test_clutter_labels=np.zeros(test_clutter.shape[0])
    test_dataset=RadarDataset_allbands(test_clutter, test_clutter_labels)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    with torch.no_grad():
        for X,y in test_loader:
            X,y=X.to(device),y.to(device)
            y_hat = net(X).squeeze(dim=1)
            y_pred = (torch.sigmoid(y_hat) >= threshold).float()
            correct = (y_pred == y).sum().item()
            total = y.size(0)
        acc = correct / total  # 计算准确率
        print(f" 测试集杂波准确率 = {acc:.3f}")
def evaluate_accuracy_gpu(net,data_iter,threshold,device=None):
    """
    用测试数据集进行测试
    :param net: 训练好的模型参数
    :param data_iter: 测试数据集
    :param device: 是否使用GPU
    :return: 返回准确率
    """
    if isinstance(net,nn.Module):
        net.eval()
        if not device:
            device=next(iter(net.parameters())).device
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X,y in data_iter:
            if isinstance(X,list):
                X=[x.to(device) for x in X]
            else:
                X=X.to(device)
            y=y.to(device)
            y_hat = net(X).squeeze(dim=1)
            y_pred = (torch.sigmoid(y_hat) >= threshold).float()  # 转换为 0/1
            metric.add(d2l.accuracy(y_pred,y),y.numel())
    return metric[0]/metric[1]
# def train(net,train_iter,test_iter,num_epochs,lr,device,threshold):
#     best_acc = 0.0
#     print('training on ',device)
#     net.to(device)
#     optimizer=torch.optim.SGD(net.parameters(),lr=lr)
#     loss=nn.BCELoss()
#     animator=d2l.Animator(xlabel='epoch',xlim=[1,num_epochs],ylim=[0,1],
#     legend=['train_loss','train_acc','test_acc'])
#     timer,num_batches=d2l.Timer(),len(train_iter)
#     for epoch in range(num_epochs):
#         metric=d2l.Accumulator(3)
#         net.train()
#         for i ,(X,y) in enumerate(train_iter):
#             timer.start()
#             optimizer.zero_grad()
#             X,y=X.to(device),y.to(device)
#             y_hat=net(X).squeeze(dim=1)
#             l=loss(y_hat,y)
#             l.backward()
#             optimizer.step()
#             # 梯度裁剪（防止爆炸）
#             torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
#             with torch.no_grad():
#                 metric.add(l*X.shape[0],d2l.accuracy((y_hat>=threshold).float(),y),X.shape[0])
#             timer.stop()
#             train_l = metric[0] / metric[2]
#             train_acc = metric[1] / metric[2]
#             if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
#                 animator.add(epoch + (i + 1) / num_batches,
#                              (train_l, train_acc, None))
#         test_acc=evaluate_accuracy_gpu(net,test_iter,threshold,device)
#         animator.add(epoch + 1, (None, None, test_acc))
#         if test_acc>best_acc:
#             best_acc=test_acc
#             save_model(net,model_name='best_model')
#             print(f"最佳测试准确率更新至: {best_acc:.3f}, 模型已保存")
#         print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
#           f'test acc {test_acc:.3f}')
#         print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
#               f'on {str(device)}')
#
#     plt.savefig('accuracy.png')
#     plt.show()

def predict(weight_path,input_length,data,device=d2l.try_gpu(),threshold=0.999,bands=4):
    # 取出测试集的前 10 个样本
    X_sample, Y_sample = next(iter(data))
    X_sample = X_sample[:10].to(device)  # 取前10个样本
    Y_sample = Y_sample[:10].to(device)  # 取前10个真实标签
    net=init_model(input_length,model_num=bands)
    net.load_state_dict(torch.load(weight_path,weights_only=True))
    net.to(device)
    net.eval()
    print("模型权重已成功加载！")
    with torch.no_grad():
        Y_pred = (net(X_sample).squeeze(dim=1)>=threshold).float() # 前向传播
    print("前10个真实标签: ", Y_sample.cpu().numpy())
    print("前10个预测结果: ", Y_pred.cpu().numpy())
    # 可视化（适用于时序数据）
    if bands==1:
        plt.figure(figsize=(10, 4))
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            plt.plot(X_sample[i].cpu().numpy().squeeze())  # 画出数据
            plt.title(f"Pred: {Y_pred[i].item()}\nLabel: {Y_sample[i].item()}")
            plt.axis("off")  # 关闭坐标轴
        plt.tight_layout()
        plt.savefig('predict_img_oneband.png')
        plt.show()
    elif bands==4:
        # 画图（适用于四个波段的数据）
        plt.figure(figsize=(12, 6))
        colors = ['r', 'g', 'b', 'y']  # 定义四个颜色，分别代表 4 个波段
        labels = ['HH', 'HV', 'VH', 'VV']  # 对应的波段名称

        for i in range(10):  # 遍历 10 个样本
            plt.subplot(2, 5, i + 1)  # 2 行 5 列的子图
            for band in range(bands):
                plt.plot(X_sample[i, band].cpu().numpy(), color=colors[band], label=labels[band])
            plt.title(f"Pred: {Y_pred[i].item()} | Label: {Y_sample[i].item()}")
            plt.legend()  # 添加图例
            plt.axis("off")  # 关闭坐标轴
        plt.tight_layout()
        plt.savefig('predict_img_allbands.png')
        plt.show()
from Dataset import *

def train_weights_save():
    set_seed(42)
    train_loader, test_loaders = get_dataloaders("train_data.h5", "test_data.h5", batch_size=64)
    train_iter, test_iter = train_loader, test_loaders['SNR_20']
    # 初始化模型
    device = d2l.try_gpu()
    # 多通道并行卷积，网络训练并且保存模型，画出pd，pf曲线
    net = init_model(input_length=210, model_num=4)
    print(net)
    train_no_sigmoid(net=net,model_name='best_cnn_mul.pth',train_iter=train_iter,
                     test_iter=test_iter,num_epochs=10,lr=0.05,
                     device=device,threshold=0.999)
    # 简单的基层卷积，。。。。。。。
    net = init_model(input_length=210, model_num=2)
    print(net)
    train_no_sigmoid(net=net,model_name='best_cnn_easy.pth',train_iter=train_iter,
                     test_iter=test_iter,num_epochs=10,lr=0.08,
                     device=device,threshold=0.999)
    # VGG11，。。。。。。。
    net = init_model(input_length=210, model_num=3)
    print(net)
    train_no_sigmoid(net=net,model_name='best_VGG11.pth',train_iter=train_iter,
                     test_iter=test_iter,num_epochs=10,lr=0.05,
                     device=device,threshold=0.999)
    # Resnet18 ....
    net = init_model(input_length=210, model_num=1)
    print(net)
    train_no_sigmoid(net=net, model_name='best_Resnet18.pth', train_iter=train_iter,
                     test_iter=test_iter, num_epochs=10,
                     lr=0.05, device=device, threshold=0.999)
def set_seed(seed=42):
    random.seed(seed)  # 固定 Python 内置 random
    np.random.seed(seed)  # 固定 NumPy
    torch.manual_seed(seed)  # 固定 PyTorch CPU
    torch.cuda.manual_seed(seed)  # 固定 PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU，确保所有都固定
    torch.backends.cudnn.deterministic = True  # 让CUDNN用确定性计算方式
    torch.backends.cudnn.benchmark = False  # 禁用自动优化算法

def write_txt_pd_pfa_all():
    set_seed(42)
    train_loader, test_loaders = get_dataloaders("train_data.h5", "test_data.h5", batch_size=64)
    predict_scr_pd_pfa(weight_path='./saved_models/best_cnn_easy.pth', model_num=2,
                       pic_name='scr_cnn_easy_pd_pfa',input_length=210,
                       test_loaders=test_loaders, device=d2l.try_gpu())
    predict_scr_pd_pfa(weight_path='./saved_models/best_cnn_mul.pth', model_num=4,
                       pic_name='scr_cnn_mul_pd_pfa',input_length=210,
                       test_loaders=test_loaders, device=d2l.try_gpu())
    predict_scr_pd_pfa(weight_path='./saved_models/best_VGG11.pth', model_num=3,
                       pic_name='scr_vgg11_pd_pfa',input_length=210,
                       test_loaders=test_loaders, device=d2l.try_gpu())
    predict_scr_pd_pfa(weight_path='./saved_models/best_Resnet18.pth', model_num=1,
                       pic_name='scr_Resnet18_pd_pfa',input_length=210,
                       test_loaders=test_loaders, device=d2l.try_gpu())
def plot_comparasion_from_txt(txt_path):
    pd={}
    with open(txt_path, 'r')as f:
        content=f.read()
    # 按模型分割数据
    models_data = content.split('\n\n')
    for model_data in models_data:
        lines = model_data.split('\n')
        model_name=lines[0].strip(':')
        pd[model_name] = []  # 初始化该模型的 pd 列表
        for line in lines[1:]:
            # 处理 pd 数据
            if "pd:" in line:
                pd_values = line.split("pd:")[1].strip().split(",")  # 获取 `pd` 后面的数据
                pd_values = [float(value) for value in pd_values]  # 转换为浮点数
                pd[model_name] = pd_values  # 存入字典

    print(pd)
    num_points = len(pd['cnn_easy'])  # 取出任意模型的 pd 数据长度
    snr_db = np.linspace(-10, 20, num_points)  # 生成等间距的 SNR 值
    for model_name,values in pd.items():
        if values == [] or model_name == []:
            break
        print(model_name)
        print(values)
        plt.plot(snr_db,values, label=model_name)
    # 添加图例
    plt.legend()

    # 添加标题和轴标签
    plt.title('PD Comparison')
    plt.xlabel('SNR (dB)')
    plt.ylabel('PD Value')
    plt.savefig('pd_comparison.png')
    # 显示图表
    plt.show()

if __name__ == '__main__':
    # write_txt_pd_pfa()
    txt_path='results_pd_pfa.txt'
    plot_comparasion_from_txt(txt_path)
    # set_seed(42)
    # train_loader, test_loaders = get_dataloaders("train_data.h5", "test_data.h5", batch_size=64)
    # train_iter, test_iter = train_loader, test_loaders['SNR_20']
    # # 初始化模型
    # device = d2l.try_gpu()
    # # # 多通道并行卷积，网络训练并且保存模型，画出pd，pf曲线
    # # net = init_model(input_length=210, model_num=4)
    # # print(net)
    # # train_no_sigmoid(net=net, model_name='best_cnn_mul.pth', train_iter=train_iter, test_iter=test_iter, num_epochs=10,
    # #                  lr=0.05, device=device, threshold=0.999)
    # # cnn_mul_pdlist,cnn_mul_pfalist=predict_scr_pd_pfa(weight_path='./saved_models/best_cnn_mul.pth', model_num=4,pic_name='scr_cnn_mul_pd_pfa',input_length=210, test_loaders=test_loaders, device=d2l.try_gpu())
    # # 简单的基层卷积，。。。。。。。
    # net = init_model(input_length=210, model_num=2)
    # print(net)
    # # 0.08 效果远好于0.05
    # train_no_sigmoid(net=net, model_name='best_cnn_easy.pth', train_iter=train_iter, test_iter=test_iter, num_epochs=10,
    #                  lr=0.08, device=device, threshold=0.999)
    # predict_scr_pd_pfa(weight_path='./saved_models/best_cnn_easy.pth', model_num=2,pic_name='scr_cnn_easy_pd_pfa',input_length=210, test_loaders=test_loaders, device=d2l.try_gpu())
    # # VGG11，。。。。。。。
    # net = init_model(input_length=210, model_num=3)
    # print(net)
    # train_no_sigmoid(net=net, model_name='best_VGG11.pth', train_iter=train_iter, test_iter=test_iter, num_epochs=10,
    #                  lr=0.05, device=device, threshold=0.999)
    # vgg11_pdlist,vgg11_pfalist=predict_scr_pd_pfa(weight_path='./saved_models/best_VGG11.pth', model_num=3,pic_name='scr_vgg11_pd_pfa',input_length=210, test_loaders=test_loaders, device=d2l.try_gpu())
    # Resnet18 ....
    # net = init_model(input_length=210, model_num=1)
    # print(net)
    # train_no_sigmoid(net=net, model_name='best_Resnet18.pth', train_iter=train_iter, test_iter=test_iter, num_epochs=10,
    #                  lr=0.05, device=device, threshold=0.999)
    # resnet18_pdlist, resnet18_pfalist = predict_scr_pd_pfa(weight_path='./saved_models/best_Resnet18.pth', model_num=1,
    #                                                  pic_name='scr_Resnet18_pd_pfa', input_length=210,
    #                                                  test_loaders=test_loaders, device=d2l.try_gpu())






