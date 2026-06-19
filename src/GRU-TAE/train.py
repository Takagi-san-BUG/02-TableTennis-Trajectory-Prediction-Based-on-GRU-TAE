import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from random import randint
import copy
import config
from tqdm import tqdm
from model import GRU_TAE
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# ---------- 1. 数据加载函数 ----------
def load_training_data():
    """加载原始 pickle 数据，并转换为 (N, 200, 3, 1) 格式"""
    data_path = config.RAW_DATA_DIR / "simulated_dataset.pkl"
    with open(data_path, 'rb') as f:
        saved_data = pickle.load(f)
    # saved_data 预期形状: (num_samples, 200, 3) ，即每条轨迹200个时间点，每点3维坐标
    # 提取坐标部分 (x, y, z)，忽略时间戳 t
    X = saved_data[:, :, 0:3]                    # (num_samples, 200, 3)
    # 增加一个通道维度，形状变为 (num_samples, 200, 3, 1)
    X = np.expand_dims(X, axis=-1)
    # 为了兼容原代码，保留 pos 和 t（但本训练不需要，仅作示例）
    pos = [list(map(lambda p: p[0:3], traj)) for traj in saved_data]
    t = [list(map(lambda p: p[3], traj)) for traj in saved_data]
    return pos, t, X

def add_noise(X):
    """向数据添加高斯噪声，并截断，增强鲁棒性"""
    if not config.add_noise:
        return X
    noise = np.random.normal(0, config.noise_std, size=X.shape)
    noise = np.clip(noise, -config.noise_clip, config.noise_clip)
    return X + noise

def split_train_val(X):
    """按比例划分训练集和验证集"""
    n = len(X)
    n_train = int(round(n * (1 - config.validation_size)))
    return X[:n_train], X[n_train:]

def create_sparse_data(x_train):
    """
    从训练集中随机生成稀疏样本（部分观测轨迹）。
    逻辑：随机选一条轨迹，随机截断，将截断后的坐标置0，并附加掩码 (1=观测, 0=缺失)。
    返回: x_sparse (NB_DATA, 200, 4), y_sparse (NB_DATA, 200, 3)
    """
    x_sparse_list = []
    y_sparse_list = []
    n_traj = x_train.shape[0]
    # 生成 NB_DATA 个稀疏样本
    for _ in range(config.nb_data):
        idx = randint(0, n_traj - 1)              # 随机选一条原始轨迹
        traj = copy.deepcopy(x_train[idx])       # (200, 3, 1) 深拷贝避免修改原始数据
        y = traj.copy()                          # 完整轨迹作为目标输出
        
        cut = randint(20, 200)                   # 随机截断点 (20~200)
        traj[cut:200, :, :] = 0                  # 截断部分坐标置零
        
        # 构造观测掩码 (200, 1)，有效部分为1，缺失部分为0
        mask = np.ones((200, 1))
        mask[cut:200, :] = 0
        
        # 去掉通道维度，变为 (200, 3)
        coords = traj.squeeze(axis=-1)
        # 拼接坐标和掩码，得到 (200, 4) 作为模型输入
        sparse = np.concatenate([coords, mask], axis=1)
        
        x_sparse_list.append(sparse)
        y_sparse_list.append(y.squeeze(axis=-1))  # 目标只需 (200, 3)
    
    return np.array(x_sparse_list), np.array(y_sparse_list)

# ---------- 2. 主训练流程 ----------
if __name__ == "__main__":
    print("正在加载数据...")
    pos, t, X = load_training_data()
    
    # 添加噪声（如果配置为 True）
    X = add_noise(X)
    
    # 划分训练集和验证集（验证集暂未用于训练，仅兼容原代码逻辑）
    x_train, x_val = split_train_val(X)
    # 只取前 NB_TRAJ_TRAIN 条轨迹用于生成稀疏样本
    x_train = x_train[:config.nb_traj_train]
    
    print("生成稀疏训练数据...")
    x_sparse, y_sparse = create_sparse_data(x_train)
    
    # 根据配置中的 length_pred 和 step_pred 对序列进行截取和降采样
    # 新序列长度 = length_pred // step_pred
    new_seq_len = config.length_pred // config.step_pred
    # 采样: 从第0个到 length_pred (不包含)，步长为 step_pred
    x_sparse = x_sparse[:, 0:config.length_pred:config.step_pred, :]  # (NB_DATA, new_seq_len, 4)
    y_sparse = y_sparse[:, 0:config.length_pred:config.step_pred, :]  # (NB_DATA, new_seq_len, 3)
    
    # 转换为 PyTorch Tensor
    x_tensor = torch.tensor(x_sparse, dtype=torch.float32)
    y_tensor = torch.tensor(y_sparse, dtype=torch.float32)
    
    # 封装为 Dataset 和 DataLoader
    dataset = TensorDataset(x_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # 初始化模型（传入采样后的序列长度）
    model = GRU_TAE(seq_len=new_seq_len)
    # 损失函数：均方误差（MSE），与原始 Keras 一致
    criterion = nn.MSELoss()
    # 优化器：Adam，使用配置的学习率
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # 将模型移动到指定设备（GPU / CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 自动选择设备
    model.to(device)
    print(f"使用设备: {device}")
    
    # ---------- 训练循环 ----------
    best_val_loss = float('inf')
    patience = 10
    no_improve = 0
    save_counter = 1  # 序号从1开始

    print("开始训练...")
    for epoch in range(config.epochs):
        model.train()                          # 切换至训练模式
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]")
        for batch_x, batch_y in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()              # 清空梯度
            output = model(batch_x)            # 前向传播
            loss = criterion(output, batch_y)  # 计算损失
            loss.backward()                    # 反向传播
            optimizer.step()                   # 更新参数
            
            total_loss += loss.item() * batch_x.size(0)  # 累加批次损失（乘以批次大小）
        
        avg_loss = total_loss / len(train_loader.dataset)  # 平均损失
        print(f"Epoch {epoch+1}/{config.epochs}, Loss: {avg_loss:.6f}")
    
        # ---------- 判断与保存 ----------
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            no_improve = 0
            # 保存模型，序号递增
            model_path = config.MODELS_DIR / f"model_{save_counter}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"  ✅ 验证损失下降，保存为 {model_path.name}")
            save_counter += 1
        else:
            no_improve += 1
            print(f"  ⚠️ 验证损失未改善，连续 {no_improve} 个epoch")
            if no_improve >= patience:
                print(f"🛑 早停触发，停止训练")
                break