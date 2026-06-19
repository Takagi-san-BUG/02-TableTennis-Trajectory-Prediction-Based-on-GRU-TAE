from pathlib import Path
from dataclasses import dataclass

ROOT_DIR = Path(__file__).parent.parent.parent
# 路径
VIDEO_DATA_DIR = ROOT_DIR / "data" / "video"  
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "process"
LOGS_DIR = ROOT_DIR / "logs"
MODELS_DIR = ROOT_DIR / "models"


# 1.数据部分
# 序列处理参数
length_pred: int = 200                            # 训练/预测时截取的序列长度（原LENGTH_PRED）
step_pred: int = 1                                # 采样步长（原STEP_PRED），步长>1会降采样
# 稀疏数据生成参数
nb_data: int = 10000                              # 生成的稀疏样本数量（原NB_DATA）,即训练及大小
nb_traj_train: int = 1900                        # 用于生成稀疏样本的原始轨迹数量（原NB_TRAJ_TRAIN）
# 数据增强与划分
validation_size: float = 0.1                     # 验证集占比
add_noise: bool = True                           # 是否添加高斯噪声
noise_std: float = 0.002                         # 噪声标准差
noise_clip: float = 0.01                         # 噪声截断范围

# 2.模型结构
input_features: int = 4                          # 输入特征数 (3个坐标 + 1个观测掩码)
output_features: int = 3                         # 输出特征数 (3个坐标重建)
# 潜在空间与隐藏层维度
latent_dim: int = 16                              # 瓶颈层维度
gru_hidden_dim: int = 64                          # GRU第一层的隐藏单元数
decoder_hidden_dim: int = 128                     # 解码器全连接层的隐藏单元数

# 3.训练参数
learning_rate: float = 0.005                      # 学习率
batch_size: int = 128                            # 批次大小
epochs: int = 50                                 # 训练轮数
 
# 4.预测参数
nb_obs: int = 60                               # 观测序列长度,seq_len-nb_obs就为预测序列长度，nb_obs越小，预测序列越长，任务难度越高



if __name__ == "__main__":
    print("配置刷新完成")