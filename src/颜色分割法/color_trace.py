import cv2
import numpy as np
import math
import config
"""
先利用HSV 颜色空间阈值分割，把画面中黄色乒乓球从背景里抠出来（颜色掩码）；
再用形态学开闭运算去除噪点干扰；
灰度二值化后通过轮廓提取定位目标物体；
借助最小外接圆、椭圆拟合获取乒乓球位置、半径、面积等信息，实现目标定位跟踪。
    
"""
# 在图片img的(x,y)坐标位置绘制红色 X 标记，用来标记乒乓球圆心
def draw_mark_X(img, x, y, width=6, color=(0, 0, 255), penWid=2):
    cv2.line(img, (int(x - width), int(y - width)), (int(x + width), int(y + width)), color, penWid)
    cv2.line(img, (int(x - width), int(y + width)), (int(x + width), int(y - width)), color, penWid)

# 设置HSV跟踪颜色范围，黄色乒乓球的 HSV 上下阈值
MIN_HSV = np.array([4, 180, 156])
MAX_HSV = np.array([32, 255, 255])
# 白色
#MIN_HSV = np.array([0, 0, 221])  # 可能需要根据具体环境调整
#MAX_HSV = np.array([180, 30, 255])
# 创建 3×3 全 1 卷积核，用于形态学操作
KERNEL = np.ones((3, 3), np.uint8)
FONT = cv2.FONT_HERSHEY_SIMPLEX
MIN_CONTOUR_AREA = 50  # 过滤小噪点轮廓最小面积
BLUR_KSIZE = (5, 5)    # 高斯模糊核大小

# 使用测试视频
video_path = str(config.VIDEO_DATA_DIR / "pingpong2.mp4")
print("当前视频完整路径：", video_path)
# video_path = 0   # 使用本机摄像头
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
if not cap.isOpened():
    raise FileNotFoundError(f"视频文件打开失败：{video_path}")
# 修正延时计算，标准应该1000毫秒
delay = int(1000 / fps)

while cap.isOpened():
    # 逐帧读取
    ret, frame = cap.read()
    # 读取失败
    if not ret:
        break
    

    # 高斯模糊降噪后把摄像头默认的 BGR 图像转换成 HSV 图像
    blur_frame = cv2.GaussianBlur(frame, BLUR_KSIZE, 0)
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 生成掩码二值图PingPangMask，提取颜色范围内图像区域
    PingPangMask = cv2.inRange(img_hsv, MIN_HSV, MAX_HSV)
    # 按位与运算，利用掩码只保留黄色乒乓球区域，其他区域全部变黑
    img_pipa = cv2.bitwise_and(frame, frame, mask=PingPangMask)


    
    # 开运算（先腐蚀后膨胀）去噪
    img_kai = cv2.morphologyEx(img_pipa, cv2.MORPH_OPEN, KERNEL)
    # 转灰度图
    gray_img = cv2.cvtColor(img_kai, cv2.COLOR_BGR2GRAY)
    # 二值化处理：亮度大于 63 的像素置为 255 白色，小于等于 63 置 0 黑
    thr, img_bin = cv2.threshold(gray_img, 63, 255, cv2.THRESH_BINARY)
    print('自动阈值thr:', thr)
    
    
    #  从二值图中提取白色物体轮廓
    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 1:
        continue
    # 按轮廓面积从大到小排序，只取最大物体
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    target_contour = contours[0]
    # 过滤过小噪点轮廓
    if cv2.contourArea(target_contour) < MIN_CONTOUR_AREA:
        continue
    # 椭圆拟合至少需要 5 个轮廓点，轮廓点太少直接跳过
    if len(target_contour) < 5:
        continue
 
    print('len(contours):', len(contours))
    print('Points', len(target_contour))
    # print('contours:', contours)
    print('hierarchy', hierarchy)
    
    # 画绿色轮廓线
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    #  画红色轮廓最小外接圆
    (x, y), radius = cv2.minEnclosingCircle(target_contour)
    center = (int(x), int(y))
    radius = int(radius)
    cv2.circle(frame, center, radius, (0, 0, 255), 1)
    draw_mark_X(frame, int(x), int(y))

    #  根据轮廓最小二乘拟合椭圆，画白色最优拟合椭圆
    ellipse = cv2.fitEllipse(target_contour)  # 最优拟合椭圆
    cv2.ellipse(frame, ellipse, (255, 255, 255), 1)
    #  计算最小外接圆面积
    area = math.pi * radius * radius
    

    #  显示圆心X Y 坐标，半径 R， 和 圆面积
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_info = f"X:{x:.1f}  Y:{y:.1f}  R:{radius}  Area:{area:.1f}"
    cv2.putText(frame, text_info, (10, 30), font, 1, (0, 255, 0), 1)

    #  显示图像帧
    cv2.imshow('PingPangTrace', frame)
    c = cv2.waitKey(int(100 / fps))
    # c = cv2.waitKey(200)
    #  按ESC键退出
    if c == 27:  # ESC
        break
cap.release()
cv2.destroyAllWindows()
print('Exit video read.')
