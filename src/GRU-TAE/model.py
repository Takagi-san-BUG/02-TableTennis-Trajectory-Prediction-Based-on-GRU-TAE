import torch
import torch.nn as nn
import config
import random
import numpy as np
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class GRU_TAE(nn.Module):
    """
    基于 GRU 的自编码器，用于轨迹补全。
    输入: (batch, seq_len, input_features)  -> 前3维为坐标，第4维为观测掩码
    输出: (batch, seq_len, output_features) -> 重建的三维坐标
    """
    def __init__(self, seq_len: int):
        super(GRU_TAE, self).__init__()
        # 从配置中读取模型结构参数
        self.seq_len = seq_len
        self.latent_dim = config.latent_dim
        input_feat = config.input_features
        output_feat = config.output_features
        gru_hidden = config.gru_hidden_dim
        decoder_hidden = config.decoder_hidden_dim

        # ----- 编码器 (Encoder) -----
        # 第一层 GRU：输入特征 -> GRU隐藏单元，return_sequences=True（返回完整序列）
        self.gru1 = nn.GRU(input_feat, gru_hidden, batch_first=True)
        # 第二层 GRU：隐藏单元 -> 潜在空间维度，return_sequences=False（只返回最后一个时间步）
        self.gru2 = nn.GRU(gru_hidden, self.latent_dim, batch_first=True)

        # ----- 解码器 (Decoder) -----
        # 将潜在向量升维到 decoder_hidden，然后映射到 seq_len * output_features
        self.fc1 = nn.Linear(self.latent_dim, decoder_hidden)
        self.fc2 = nn.Linear(decoder_hidden, seq_len * output_feat)
        
        # 激活函数
        self.elu = nn.ELU()

    def encode(self, x):
        """
        编码器：将输入序列压缩为潜在向量 z
        输入 x: (batch, seq_len, input_features)
        返回 z: (batch, latent_dim)
        """
        # 经过第一层 GRU，输出形状 (batch, seq_len, gru_hidden)
        out, _ = self.gru1(x)       
        # 经过第二层 GRU，out 形状 (batch, seq_len, latent_dim)，但这里只需要最后一步隐状态
        _, h_n = self.gru2(out)      # h_n 形状: (1, batch, latent_dim) 因为只有一层
        # 去掉第0维（层数维度），得到 (batch, latent_dim)
        z = h_n.squeeze(0)
        return z

    def decode(self, z):
        """
        解码器：从潜在向量 z 重建序列
        输入 z: (batch, latent_dim)
        返回 recon: (batch, seq_len, output_features)
        """
        # 全连接层1 + ELU 激活
        x = self.elu(self.fc1(z))    # (batch, decoder_hidden)
        # 全连接层2，输出长度 = seq_len * output_features
        x = self.fc2(x)              # (batch, seq_len * output_features)
        # 重塑为序列形式 (batch, seq_len, output_features)
        recon = x.view(-1, self.seq_len, config.output_features)
        return recon

    def forward(self, x):
        """前向传播：编码 + 解码"""
        z = self.encode(x)
        recon = self.decode(z)
        return recon