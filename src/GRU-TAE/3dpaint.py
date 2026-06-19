import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import config
# ---------- 中文字体支持 ----------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取Excel数据
def read_trajectory_data(file_path):
    """从Excel文件读取轨迹数据"""
    df = pd.read_excel(file_path)
    print(f"成功读取数据，共 {len(df)} 个数据点")
    print("数据样例:\n", df.head())
    return df[['X', 'Y', 'Z']].values


def plot_interactive_3d(data):
    """生成可交互的3D轨迹图"""
    # 启用交互模式
    plt.ion()

    # 创建3D画布
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 提取坐标数据
    x, y, z = data[:, 0], data[:, 1], data[:, 2]

    # 绘制轨迹
    ax.plot(x, y, z,
            marker='o',
            markersize=4,
            linestyle='-',
            linewidth=1,
            color='#e76f51',
            alpha=0.8)

    # 设置等比例坐标轴
    set_equal_axes(ax, x, y, z)

    # 设置标签
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    plt.title("Interactive 3D Trajectory\n(鼠标拖动旋转视角)", pad=20)

    # 关闭交互模式并保持窗口
    plt.ioff()
    plt.show()


def set_equal_axes(ax, x, y, z):
    """设置等比例三维坐标轴"""
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
# 主程序
if __name__ == "__main__":
    # 参数设置

    # 数据读取（替换为实际文件路径）
    excel_path = str(config.PROCESSED_DATA_DIR / "trajectory_part1/0001.xlsx")
    trajectory = read_trajectory_data(excel_path)

    # 生成交互式可视化
    plot_interactive_3d(trajectory)












