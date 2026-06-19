import cv2
import numpy as np
import config

"""
    HSV 颜色阈值目标检测 + 帧间差分运动检测 + Lucas-Kanade 金字塔稀疏光流（LK 光流）跟踪
先用 HSV 颜色分割把黄色乒乓球区域提取出来
用帧差法过滤静止背景，只保留运动的乒乓球区域，消除静止黄色物体干扰
轮廓筛选得到乒乓球候选区域
在球区域提取 Shi-Tomasi 角点，使用LK 金字塔稀疏光流跟踪特征点，修正小球中心点位置
记录每帧球坐标，绘制历史运动轨迹
"""

kernel = np.ones((5, 5), np.uint8)  # 形态学操作核
# 设置HSV跟踪颜色范围，黄色或橙色
min_pipa_hsv = np.array([4, 180, 156])
max_pipa_hsv = np.array([32, 255, 255])
# Shi-Tomasi角点参数
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
# LK光流参数
lk_params = dict(winSize=(15, 15), maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def detect_and_track_ball(frame, prev_mask, trajectory, prev_gray, max_distance=20):


    # 当前帧转灰度图：用于后续角点提取、LK 光流计算
    # 当前帧转 HSV，根据阈值生成二值掩码：黄色区域白色 255，其余黑色 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, min_pipa_hsv, max_pipa_hsv)
    found = False
    
    # 第一帧保存初始颜色掩码，用于后续帧差计算
    if prev_mask is None:
        prev_mask = mask.copy()

    # 对帧差进行高斯模糊以减少噪声
    # 帧间差分，当前掩码 - 上一帧掩码，只有发生像素变化的区域（运动区域）才会出现白色，过滤静止背景里的黄色干扰物
    frame_diff = cv2.absdiff(mask, prev_mask)
    blurred = cv2.GaussianBlur(frame_diff, (9, 9), 0)  # 高斯模糊
    # 二值化：差值大于 25 的像素置白，其余变黑，得到运动区域掩码
    _, thresh = cv2.threshold(blurred, 25, 255, cv2.THRESH_BINARY)
    
    # 开运算：先腐蚀后膨胀，消除帧差产生的细小噪点，平滑运动目标轮廓
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)  # 去噪

    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    center = None
    # 筛选面积最大的轮廓作为乒乓球，规避小噪声轮廓误检
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        # 获取轮廓最小外接圆的圆心、半径
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        if M["m00"] > 0:  # 确保面积足够大
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # 条件：球半径大于 10 像素；首次检测 或者 当前质心和上一个轨迹点距离大于max_distance才开启光流跟踪。
            #if radius > 10 and (len(trajectory) == 0 or np.linalg.norm(np.array(center) - np.array(trajectory[-1])) < max_distance):
            if radius > 20:
                found = True
                # 在检测到的区域内提取特征点
                # Shi-Tomasi 角点检测：在当前整幅灰度图最多提取 100 个高质量特征点；
                # 创建空白掩码，仅在乒乓球轮廓区域允许提取特征点
                feature_mask = np.zeros_like(gray)
                cv2.drawContours(feature_mask, [c], -1, 255, -1)
                
                p0 = cv2.goodFeaturesToTrack(gray, mask=feature_mask, **feature_params)

                # LK 金字塔稀疏光流：根据上一帧特征点p0，追踪这些点在当前帧的新位置p1
                if p0 is not None and len(p0) > 0 and prev_gray is not None:
                    # 计算光流

                    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)

                    # 选择好的点
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]

                    # 遍历所有成功跟踪的特征点，找到距离乒乓球质心最近的特征点，用该特征点坐标覆盖原有质心，实现位置修正。
                    for i, (new, old) in enumerate(zip(good_new, good_old)):
                        a, b = new.ravel()
                        c, d = old.ravel()
                        distance = np.linalg.norm(np.array([a, b]) - np.array(center))
                        if distance < max_distance:  # 检查是否靠近之前检测到的中心
                            center = (int(a), int(b))
                            break

                trajectory.append(center)
                # # 限制最多保存150个轨迹点，避免内存持续上涨
                
                max_trajectory_len = 100
                if len(trajectory) > max_trajectory_len:
                    trajectory.pop(0)
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                
    if not found:
        trajectory.append(None)
        if len(trajectory) > 150:
            trajectory.pop(0)
            
    # 绘制轨迹
    for i in range(1, len(trajectory)):
        if trajectory[i - 1] is not None and trajectory[i] is not None:
            thickness = int(np.sqrt(len(trajectory) / float(i + 1)) * 2.5)
            cv2.line(frame, trajectory[i - 1], trajectory[i], (0, 0, 255), thickness)

    return frame,  mask, trajectory, gray  # 返回当前帧的灰度图像供下一次迭代使用


# 主程序开始
# 使用测试视频
video_path = str(config.VIDEO_DATA_DIR / "pingpong2.mp4")
print("当前视频完整路径：", video_path)
cap = cv2.VideoCapture(video_path)  # 替换为你的本地视频路径

prev_frame = None
prev_mask = None
trajectory = []
prev_gray = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    # 将当前帧转换为灰度图
    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    try:
        frame, prev_mask, trajectory, prev_gray = detect_and_track_ball(
            frame, prev_mask, trajectory, prev_gray)
        prev_gray = current_gray.copy()

    except Exception as e:
        print(f"Error during detection and tracking: {e}")
        continue

    cv2.imshow('Frame', frame)

    k = cv2.waitKey(5) & 0xff
    if k == 27:  # ESC退出
        break

cap.release()
cv2.destroyAllWindows()