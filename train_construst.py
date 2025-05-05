from os import makedirs
import torch.optim
from train import *
from model_construst import *

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features: [batch_size, proj_dim]
        labels: [batch_size]
        """
        device = features.device
        batch_size = features.shape[0]

        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(device)

        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T), self.temperature
        )

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        # loss
        loss = -mean_log_prob_pos
        loss = loss.mean()
        return loss
def init_model(input_length=210, model_num=2):
    if model_num == 1:
        model = CNN_easy(input_length)
    elif model_num == 2:
        classifier = nn.Sequential(
            nn.Linear(512, 1)
        )
        model = nn.Sequential(
            HRRP_ResNet_backbone().features,
            classifier)
    elif model_num == 3:
        classifier = nn.Sequential(
            nn.Linear(128 * (input_length // 32), 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )
        model=nn.Sequential(
            HRRP_VGG11_backbone(input_length=input_length).features,
            classifier
        )
        # model = HRRP_VGG11(input_length)
    return model
def train_backbone_contrastive(
    backbone_model,                    # 传入Backbone（比如 HRRP_ResNet()）
    train_loader,                      # 传入训练DataLoader
    input_length=210,                  # 输入长度（默认210）
    feature_dim=512,                   # Backbone输出特征维度
    proj_dim=128,                      # 投影头输出维度
    temperature=0.1,                   # SupCon温度超参数
    lr=0.008,                          # 学习率
    epochs=20,                         # 训练轮数
    save_path='SCL/backbone_resnet18.pth',       # 保存的路径
    seed=42                             # 随机种子
):
    # 设置随机种子
    set_seed(seed)

    device = d2l.try_gpu()
    # 定义SupCon模型
    class SupConModel(nn.Module):
        def __init__(self, backbone, feature_dim, proj_dim):
            super(SupConModel, self).__init__()
            self.backbone = backbone
            self.projection_head = ProjectionHead(feature_dim, proj_dim)

        def forward(self, x):
            features = self.backbone(x)
            projections = self.projection_head(features)
            return projections

    model = SupConModel(backbone_model, feature_dim, proj_dim)
    model = model.to(device)

    # 损失函数和优化器
    criterion = SupConLoss(temperature=temperature)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 开始训练
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (X, y) in enumerate(train_loader):
            inputs, labels = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    # 保存backbone参数
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.backbone.state_dict(), save_path)
    print(f"✅ Backbone saved at {save_path}")

def finetune_with_BCE_loss(
    backbone_model,                # Backbone模型 (如 HRRP_ResNet)
    train_loader,                  # 训练集loader
    val_loader,                    # 验证集loader
    input_length=210,
    model_name='./SCL/Resnet18_finetuned_model.pth', # 保存模型路径
    num_epochs=10,                 # 训练轮数
    lr=0.001,                      # 学习率
    snr_range=None,                # SNR范围 (日志显示用)
    device=d2l.try_gpu(),           # 设备
    threshold=0.999,                 # sigmoid后判决阈值
    unfreeze_backbone=False        # 是否解冻backbone
):
    # 冻结
    if not unfreeze_backbone:
        for param in backbone_model.parameters():
            param.requires_grad = False
    # ResNet18
    # # 新加分类器（你原来是512 -> 1）
    # classifier = nn.Sequential(
    #     nn.Linear(512, 1)
    # )
    #
    # model = nn.Sequential(
    #     backbone_model.features,  # 只要提取特征部分
    #     classifier
    # )
    # VGG11
    classifier = nn.Sequential(
        nn.Linear(128 * (input_length // 32), 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 1),
    )
    model=nn.Sequential(
        backbone_model.features,
        classifier
    )
    model.to(device)

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    pos_weight = torch.tensor([1.0]).to(device)  # 根据需要调整
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 日志
    best_acc = 0.0
    snr_str = f"SNR {snr_range[0]} to {snr_range[1]}" if snr_range else "All SNR"
    print(f"Training on {device} for {snr_str}")

    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train_loss', 'train_acc', 'val_acc'])
    timer, num_batches = d2l.Timer(), len(train_loader)

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # 累加loss、准确率、样本数
        model.train()
        for i, (X, y) in enumerate(train_loader):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            logits = model(X).squeeze(dim=1)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            with torch.no_grad():
                y_pred = (torch.sigmoid(logits) >= threshold).float()
                metric.add(loss * X.shape[0], d2l.accuracy(y_pred, y), X.shape[0])
            timer.stop()

            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (train_l, train_acc, None))

        val_acc = evaluate_accuracy_gpu(model, val_loader, threshold, device)
        animator.add(epoch + 1, (None, None, val_acc))

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_name)
            print(f"最佳测试准确率更新至: {best_acc:.3f}, 模型已保存")

        print(f'{snr_str} Epoch {epoch + 1}: loss {train_l:.5f}, train acc {train_acc:.3f}, val acc {val_acc:.3f}')
        print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(device)}')

        if train_l < 1e-4:
            print('loss < 1e-4, stop training this stage')
            break

    plt.savefig(f'accuracy_{snr_str.replace(" ", "_")}.png')
    plt.close()


def predict_scr_pd_pfa(weight_path,model_num,clutter_loader,test_loaders,pic_name,result_txt_path,
                       device=d2l.try_gpu(),input_length=210,threshold=0.999):
    models = {
        "1": "cnn_easy",
        "3": "VGG11",
        "2": "Resnet18",
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
        cl_sum,cl_false=0,0
        for X_clutter, y_clutter in clutter_loader:
            X_clutter, y_clutter = X_clutter.to(device), y_clutter.to(device)
            y_hat_cl = net(X_clutter).squeeze(dim=1)
            y_pred = (torch.sigmoid(y_hat_cl) >= threshold).float()  # 计算预测值
            cl_sum+=y_hat_cl.size(0)
            cl_false += (y_pred != y_clutter).sum().item()
        pfa=cl_false/cl_sum
        print(pfa)
        pfa_list.append(pfa)
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
            pd_list.append(pd)

            print(f"SNR {snr}: Pd = {pd:.5f}")

    # 绘制 Pd 和 Pfa 随 SNR 变化的曲线
    plt.figure(figsize=(8, 6))
    plt.plot(snr_db, pd_list, marker='o', linestyle='-', color='b', label=f'{models[str(model_num)]}Pd')
    # plt.plot(snr_db, pfa_list, marker='s', linestyle='--', color='r', label='Pfa (False Alarm Rate)')

    plt.xlabel('SCR (dB)')
    plt.ylabel('Pd')
    plt.title('Pd vs SNR')
    plt.legend()
    plt.grid(True)
    plt.savefig(pic_name)  # 保存图片
    plt.show()
    # 将pd_list,pfa_list写入txt文件
    with open(result_txt_path, "a") as f:
        f.write(f"{models[str(model_num)]}:\n")
        f.write("pd: " + ",".join(map(str, pd_list)) + "\n")
        f.write("pfa: " + ",".join(map(str, pfa_list)) + "\n\n")

set_seed(42)
train_loaders,val_loader=get_dataloaders("train.h5", "val.h5",batch_size=128)
train_loader=train_loaders['SNR_-10_20']
input_length=210
# backbone=HRRP_VGG11_backbone(input_length=210)
# train_backbone_contrastive(backbone_model=backbone,train_loader=train_loader,lr=0.008,
#                            feature_dim=128 * (input_length // 32),save_path='./SCL/backbone_VGG11.pth')

backbone=HRRP_VGG11_backbone(input_length=210)
backbone.load_state_dict(torch.load('SCL/backbone_VGG11.pth', weights_only=True))
# lr resnet18 0.005
finetune_with_BCE_loss(backbone_model=backbone,train_loader=train_loader,val_loader=val_loader,
                       model_name='SCL/VGG11_finetuned_model.pth',  num_epochs=50,lr=0.008)
clutter_loader, test_loaders=get_dataloaders_test("test.h5",batch_size=128)
predict_scr_pd_pfa(
    weight_path='SCL/VGG11_finetuned_model.pth',
    model_num=3,
    clutter_loader=clutter_loader,
    test_loaders=test_loaders,
    pic_name='pd_vs_snr_VGG11_SCL.png',
    result_txt_path='results_pd_pfa_base_SCL.txt'
)
# feature_dim = 512
# proj_dim = 128
# model=SupConModel(input_length=210, feature_dim=feature_dim, proj_dim=proj_dim)
# model.backbone=HRRP_ResNet()
# model.projection_head = ProjectionHead(feature_dim, proj_dim)
# model.to(device)
# # 损失函数和优化器
# criterion = SupConLoss(temperature=0.1)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.008)
# # 训练过程
# epochs = 20
# for epoch in range(epochs):
#     model.train()
#     total_loss = 0
#     for i ,(X,y) in enumerate(train_loader):
#         inputs, labels =X.to(device), y.to(device)
#         optimizer.zero_grad()
#         y_pred=model(inputs)
#         loss = criterion(y_pred, labels)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")
# makedirs('SCL', exist_ok=True)
# torch.save(model.backbone.state_dict(), 'SCL/backbone_resnet18.pth')