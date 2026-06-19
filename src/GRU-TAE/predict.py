import torch
import numpy as np
import pickle
import config
from model import GRU_TAE
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# ---------- 中文字体支持 ----------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def plot_comparison_3d(true_traj, observed_traj, predicted_traj, 
                       obs_len, title="轨迹补全对比"):
    """
    绘制真实轨迹、观测部分、预测部分的3D对比图。
    
    参数:
        true_traj: 完整真实轨迹，形状 (seq_len, 3)
        observed_traj: 观测到的部分，形状 (obs_len, 3)
        predicted_traj: 模型预测的补全部分，形状 (seq_len-obs_len, 3)
        obs_len: 观测帧数（用于分段）
        title: 图标题
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. 绘制完整真实轨迹（灰色虚线，作为参考）
    ax.plot(true_traj[:, 0], true_traj[:, 1], true_traj[:, 2],
            color='gray', linestyle='--', linewidth=1.5, alpha=0.6, label='真实完整轨迹')
    
    # 2. 绘制观测部分（蓝色实线 + 圆点）
    ax.plot(observed_traj[:, 0], observed_traj[:, 1], observed_traj[:, 2],
            color='#1f77b4', marker='o', markersize=4, linewidth=2, label='观测部分')
    
    # 3. 绘制预测补全部分（红色实线 + 方块标记）
    ax.plot(predicted_traj[:, 0], predicted_traj[:, 1], predicted_traj[:, 2],
            color='#d62728', marker='s', markersize=4, linewidth=2, label='预测补全')
    
    # 设置等比例坐标轴（复用你之前的函数）
    all_points = np.vstack([true_traj, observed_traj, predicted_traj])
    set_equal_axes(ax, all_points[:, 0], all_points[:, 1], all_points[:, 2])
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()
    
    plt.show()

def set_equal_axes(ax, x, y, z):
    """保持3D坐标轴等比例"""
    max_range = np.array([x.max() - x.min(),
                          y.max() - y.min(),
                          z.max() - z.min()]).max() * 0.5
    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    mid_z = (z.max() + z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_box_aspect([1, 1, 1])
    
    
    
# ---------- 1. 加载模型 ----------
def load_model():
    """加载训练好的模型权重，并设置为评估模式"""
    # 计算模型期望的序列长度（必须与训练时一致）
    new_seq_len = config.length_pred // config.step_pred
    model = GRU_TAE(seq_len=new_seq_len)
    # 加载权重，map_location 确保在 CPU 上也能加载（如果训练在 GPU）
    model.load_state_dict(torch.load(config.MODELS_DIR/"model_28.pth", map_location=torch.device('cpu')))
    model.eval()   # 切换至评估模式（关闭 dropout 等）
    return model

# ---------- 2. 测试数据预处理 ----------
def prepare_test_data(test_raw_data, nb_obs=60):
    """
    将原始测试轨迹处理为稀疏输入（保留前 nb_obs 步，其余置零并添加掩码）
    参数:
        test_raw_data: (N, 200, 3, 1) 格式的原始数据
        nb_obs: 保留的观测步数（截断点）
    返回:
        x_test_sparse: (N, new_seq_len, 4) 稀疏输入
    """
    new_seq_len = config.length_pred // config.step_pred
    x_sparse_list = []
    
    for traj in test_raw_data:
        traj = traj.squeeze(axis=-1)          # (200, 3)
        new_traj = traj.copy()
        # 将 nb_obs 之后的坐标置零
        new_traj[nb_obs:200, :] = 0
        # 构造观测掩码
        mask = np.ones((200, 1))
        mask[nb_obs:200, :] = 0
        # 拼接坐标和掩码 -> (200, 4)
        sparse = np.concatenate([new_traj, mask], axis=1)
        # 按配置的 step_pred 进行降采样
        sparse = sparse[0:config.length_pred:config.step_pred, :]  # (new_seq_len, 4)
        x_sparse_list.append(sparse)
    
    return np.array(x_sparse_list)

if __name__ == "__main__":
    # ---------- 配置 ----------
    # 请将下面的路径改为你自己的保存目录
    SAVE_DIR = config.LOGS_DIR

    # ---------- 1. 加载模型 ----------
    model = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ---------- 2. 加载全部数据 ----------
    with open(config.RAW_DATA_DIR / "simulated_dataset.pkl", 'rb') as f:
        saved_data = pickle.load(f)
    X = saved_data[:, :, 0:3]                     # (N, 200, 3)
    X = np.expand_dims(X, axis=-1)                # (N, 200, 3, 1)
    N = X.shape[0]
    print(f"数据集共有 {N} 条轨迹")

    # ---------- 3. 批量测试集评估 ----------
    test_data = X[-20:]                           # 最后 20 条
    x_test_sparse = prepare_test_data(test_data, nb_obs=config.nb_obs)
    x_tensor = torch.tensor(x_test_sparse, dtype=torch.float32).to(device)

    with torch.no_grad():
        reconstructions = model(x_tensor)
    preds = reconstructions.cpu().numpy()         # (20, new_seq_len, 3)

    new_seq_len = config.length_pred // config.step_pred
    y_true = test_data[:, 0:config.length_pred:config.step_pred, :].squeeze(-1)
    test_mse = np.mean((preds - y_true) ** 2)
    print(f"测试集（最后20条）整体重建 MSE: {test_mse:.6f}")

    # ---------- 4. 批量保存测试集对比图 ----------
    if SAVE_DIR != "":
        print(f"正在保存测试集图片至：{SAVE_DIR}")
        for i in range(20):
            true_full = test_data[i].squeeze(-1)            # (200, 3)
            true_full_ds = true_full[0:config.length_pred:config.step_pred, :]
            sparse_input = x_test_sparse[i]                 # (new_seq_len, 4)
            mask = sparse_input[:, 3]
            obs_indices = np.where(mask > 0.5)[0]
            pred_indices = np.where(mask < 0.5)[0]
            observed_traj = sparse_input[obs_indices, :3]
            predicted_traj = preds[i][pred_indices]

            # 绘制并保存（不显示）
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(true_full_ds[:, 0], true_full_ds[:, 1], true_full_ds[:, 2],
                    color='gray', linestyle='--', linewidth=1.5, alpha=0.6, label='真实完整轨迹')
            ax.plot(observed_traj[:, 0], observed_traj[:, 1], observed_traj[:, 2],
                    color='#1f77b4', marker='o', markersize=4, linewidth=2, label='观测部分')
            ax.plot(predicted_traj[:, 0], predicted_traj[:, 1], predicted_traj[:, 2],
                    color='#d62728', marker='s', markersize=4, linewidth=2, label='预测补全')
            all_points = np.vstack([true_full_ds, observed_traj, predicted_traj])
            set_equal_axes(ax, all_points[:, 0], all_points[:, 1], all_points[:, 2])
            ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
            ax.set_title(f"测试样本 {i} (观测 {len(obs_indices)} 帧)")
            ax.legend()
            save_path = SAVE_DIR / f"test_sample_{i}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)   # 释放内存，不弹出窗口
        print("测试集图片保存完成。")

    # ---------- 5. 交互式单条轨迹可视化 ----------
    sample_idx = int(input(f"请输入要可视化的轨迹索引 (0~{N-1}): "))
    if sample_idx < 0 or sample_idx >= N:
        raise ValueError(f"索引越界！有效范围 0~{N-1}")

    single_traj = X[sample_idx:sample_idx+1]
    x_sparse_single = prepare_test_data(single_traj, nb_obs=config.nb_obs)
    x_single_tensor = torch.tensor(x_sparse_single, dtype=torch.float32).to(device)

    with torch.no_grad():
        pred_single = model(x_single_tensor)
    pred_single = pred_single.cpu().numpy()[0]

    true_full_ds = X[sample_idx, 0:config.length_pred:config.step_pred, :].squeeze(-1)
    single_mse = np.mean((pred_single - true_full_ds) ** 2)
    print(f"样本 {sample_idx} 重建 MSE: {single_mse:.6f}")

    # 可视化（显示在屏幕）
    true_full = X[sample_idx].squeeze(-1)
    true_full_ds_plot = true_full[0:config.length_pred:config.step_pred, :]
    sparse_input = x_sparse_single[0]
    mask = sparse_input[:, 3]
    obs_indices = np.where(mask > 0.5)[0]
    pred_indices = np.where(mask < 0.5)[0]
    observed_traj = sparse_input[obs_indices, :3]
    predicted_traj = pred_single[pred_indices]

    plot_comparison_3d(
        true_traj=true_full_ds_plot,
        observed_traj=observed_traj,
        predicted_traj=predicted_traj,
        obs_len=len(obs_indices),
        title=f"样本 {sample_idx} 轨迹补全效果 (观测 {len(obs_indices)} 帧, MSE={single_mse:.6f})"
    )