# <div align="center">基于GRU-TAE的乒乓球轨迹预测</div>

<div align="center">
  <p><strong>乒乓球视频目标检测 → 轨迹提取 → 基于 GRU 时序自编码器的轨迹补全与预测</strong></p>
  <p>颜色分割 · 帧差+光流 · GRU-TAE 轨迹建模</p>
</div>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Status](https://img.shields.io/badge/Status-Research%20Project-6B46C1?style=for-the-badge)

</div>

---

## 概述

本项目围绕乒乓球轨迹预测这一任务，构建了从视频目标检测到轨迹建模的完整流程，包含三条技术路线：

- 基于 HSV 颜色空间的传统视觉检测（颜色分割法、帧差法+光流法），从视频中实时提取乒乓球的二维运动轨迹
- 基于 GRU 时序自编码器（GRU-TAE）的深度学习轨迹建模，在部分观测条件下补全缺失的三维轨迹

项目在仿真数据集上训练 GRU-TAE 模型，并在真实采集的乒乓球轨迹上验证模型的泛化能力。整体工作偏向学术研究与实验探索，适合用来理解时序自编码器在轨迹补全任务上的应用。

---

## 特性

- 三种乒乓球检测与跟踪方法：纯颜色分割、颜色分割+帧差+光流联合跟踪、深度学习轨迹补全
- GRU-TAE 模型：双层 GRU 编码器 + 全连接解码器，将部分观测轨迹压缩为潜在向量后重建完整轨迹
- 支持高斯噪声数据增强，提升模型鲁棒性
- 基于 early stopping 的训练策略，自动保存最优模型
- 3D 可视化对比：真实轨迹、观测部分、预测补全部分同时展示
- 支持批量测试集评估与单条轨迹交互式可视化

---

## 仓库结构

```text
02-基于GRU-TAE的乒乓球轨迹预测/
├─ README.md
├─ data/
│  ├─ video/                        # 乒乓球比赛视频（.mp4）
│  ├─ raw/                          # 原始轨迹数据（.pkl）
│  │  ├─ simulated_dataset.pkl      # 仿真生成的 3D 轨迹（2000条, 200帧, 4维）
│  │  └─ real_balls_dataset.pkl     # 真实采集的 3D 轨迹
│  └─ process/                      # 处理后的轨迹数据
│     └─ trajectory_part1/          # Excel 格式的单条轨迹（X, Y, Z）
├─ models/                          # 训练好的模型权重（.pth）
├─ logs/                            # 测试集可视化结果（.png）
└─ src/
   ├─ 颜色分割法/                    # 纯 HSV 颜色阈值检测
   │  ├─ config.py
   │  └─ color_trace.py
   ├─ 帧差法+光流法/                  # HSV + 帧间差分 + LK 光流联合跟踪
   │  ├─ config.py
   │  └─ detect_ball_trajectory.py
   └─ GRU-TAE/                      # GRU 时序自编码器轨迹补全
      ├─ config.py                  # 路径、模型超参数、训练超参数
      ├─ model.py                   # GRU_TAE 模型定义
      ├─ train.py                   # 数据加载、稀疏样本生成、训练循环
      ├─ predict.py                 # 模型推理与 3D 可视化对比
      └─ 3dpaint.py                 # 轨迹数据 3D 交互式绘图工具
```

核心文件说明：

- `src/GRU-TAE/config.py`：路径配置、数据参数、模型结构超参数、训练超参数
- `src/GRU-TAE/model.py`：GRU_TAE 类，包含 encode / decode / forward，输入 (B, T, 4) 输出 (B, T, 3)
- `src/GRU-TAE/train.py`：从 pickle 加载数据 → 噪声增强 → 生成稀疏样本 → DataLoader → 训练 → 保存模型
- `src/GRU-TAE/predict.py`：加载模型 → 测试集评估 → 批量保存对比图 → 交互式单轨迹可视化
- `src/GRU-TAE/3dpaint.py`：读取 Excel 轨迹数据并生成可交互的 3D 轨迹图

---

## 环境

推荐使用 Conda 创建独立环境：

```powershell
conda create -n pingpong python=3.10
conda activate pingpong
pip install torch opencv-python pandas matplotlib openpyxl tqdm
```

本项目不包含 `requirements.txt`，可按需安装上述依赖。主要依赖：

- PyTorch（GPU 或 CPU 版本均可）
- OpenCV（视频读取与图像处理）
- NumPy / Pandas（数据处理）
- Matplotlib（可视化）
- tqdm（训练进度条）
- openpyxl（Excel 读写）

---

## 数据

### 仿真数据集

`data/raw/simulated_dataset.pkl` 包含 2000 条仿真生成的乒乓球三维轨迹，每条轨迹 200 个时间步，每步 4 个特征 (x, y, z, t)，形状为 (2000, 200, 4)。

数据通过物理引擎模拟乒乓球在球台上的飞行与弹跳，包含旋转、空气阻力等因素。模型训练时仅使用前 3 维坐标，第 4 维时间戳仅用于辅助分析。

### 真实采集数据

`data/raw/real_balls_dataset.pkl` 包含从真实乒乓球比赛视频中提取的三维轨迹数据。

### 处理后轨迹

`data/process/trajectory_part1/` 下为从视频中逐帧提取并处理后的轨迹，每条保存为一个 Excel 文件，包含 X、Y、Z 三列坐标。

### 视频数据

`data/video/` 下包含乒乓球比赛视频（.mp4），供颜色分割法和帧差+光流法使用。

---

## 三条技术路线

### 1. 颜色分割法

纯 HSV 颜色空间阈值分割，流程为：

高斯模糊降噪 → BGR 转 HSV → 颜色阈值生成掩码 → 形态学开运算去噪 → 灰度二值化 → 轮廓提取 → 最小外接圆/椭圆拟合定位球心

运行方式：

```powershell
python src/颜色分割法/color_trace.py
```

优点：实现简单，对黄色乒乓球检测效果好。缺点：易受光照变化和同色背景干扰。

### 2. 帧差法 + 光流法

在 HSV 颜色分割的基础上加入运动信息，流程为：

HSV 颜色掩码 → 帧间差分过滤静止背景 → 高斯模糊 + 二值化 → 轮廓筛选 → Shi-Tomasi 角点提取 → LK 金字塔稀疏光流跟踪 → 球心位置修正

运行方式：

```powershell
python src/帧差法+光流法/detect_ball_trajectory.py
```

优点：能过滤静止的黄色干扰物，光流跟踪使球心定位更稳定。缺点：计算量较大，对快速运动可能跟踪丢失。

### 3. GRU-TAE 轨迹补全

核心思路：给定部分观测的轨迹（前 N 帧有值，后续帧置零），通过时序自编码器重建完整轨迹。

**模型结构：**

- 编码器：双层 GRU，第一层将 4 维输入（3 坐标 + 1 观测掩码）映射到 64 维隐藏空间，第二层压缩为 16 维潜在向量
- 解码器：全连接网络，将潜在向量升维到 128 维，再映射为完整序列（T × 3 坐标）
- 激活函数：ELU

**训练数据构造：**

从原始轨迹中随机选取一条，随机选取截断点（20~200），截断部分坐标置零并附加观测掩码（1=观测, 0=缺失），生成 (NB_DATA, seq_len, 4) 的稀疏输入和 (NB_DATA, seq_len, 3) 的完整目标。

---

## 默认配置

当前默认超参数来自 `src/GRU-TAE/config.py`：

| 参数 | 值 | 说明 |
|---|---|---|
| `length_pred` | 200 | 序列长度 |
| `step_pred` | 1 | 采样步长 |
| `nb_data` | 10000 | 生成的稀疏样本数 |
| `nb_traj_train` | 1900 | 用于训练的原轨迹数 |
| `latent_dim` | 16 | 瓶颈层维度 |
| `gru_hidden_dim` | 64 | GRU 隐藏单元数 |
| `decoder_hidden_dim` | 128 | 解码器隐藏单元数 |
| `learning_rate` | 0.005 | 学习率 |
| `batch_size` | 128 | 批次大小 |
| `epochs` | 50 | 最大训练轮数 |
| `nb_obs` | 60 | 观测帧数（前60帧可见，后140帧需预测） |

---

## 训练

在仓库根目录运行：

```powershell
python src/GRU-TAE/train.py
```

训练过程：

- 从 `data/raw/simulated_dataset.pkl` 加载 2000 条轨迹
- 可选添加高斯噪声增强（默认 std=0.002）
- 随机生成 10000 个稀疏样本（部分观测+掩码）
- 在训练集上训练，每轮输出平均 MSE 损失
- 损失下降时自动保存模型至 `models/model_N.pth`
- 连续 10 轮无改善则触发 early stopping

---

## 推理与可视化

```powershell
python src/GRU-TAE/predict.py
```

推理流程：

- 自动加载 `models/model_28.pth`（可在代码中修改为其他 checkpoint）
- 取数据集最后 20 条轨迹作为测试集，前 60 帧为观测，后 140 帧需预测
- 输出测试集整体 MSE
- 批量生成 20 张 3D 对比图保存至 `logs/`，图中灰色虚线为真实轨迹，蓝色为观测部分，红色为模型预测补全
- 支持交互式输入轨迹索引，弹出单条轨迹的 3D 对比图窗口

3D 轨迹绘图工具：

```powershell
python src/GRU-TAE/3dpaint.py
```

读取 `data/process/trajectory_part1/` 中的 Excel 轨迹文件，生成可鼠标拖拽旋转的交互式 3D 轨迹图。

---

## 模型文件

`models/` 目录下保存了 28 个训练过程中产生的 checkpoint（`model_1.pth` ~ `model_28.pth`），按训练过程中损失下降的时刻依次保存，编号越大通常代表训练越充分（但也可能因过拟合导致泛化性能下降）。

---

## 备注

- 视频检测脚本（颜色分割法、帧差+光流法）依赖 OpenCV 的 GUI 窗口，需在有图形界面的环境下运行，按 ESC 键退出
- 仿真数据集为物理引擎生成，与真实场景存在 domain gap，模型在真实数据上的泛化效果可能有限
- 当前 GRU-TAE 解码器为纯全连接结构，未使用自回归解码，优点是推理速度快，缺点是长序列重建精度可能不足
- 模型输入的第 4 维为观测掩码，显式告知模型哪些帧是真实观测、哪些是缺失值，是轨迹补全任务的关键设计
- `data/` 目录下的 .pkl 和 .mp4 文件较大，未纳入 Git 版本管理，如需复现请自行准备数据
