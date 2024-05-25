import os
import cv2

# 定义图片目录
img_dir = './yuan'
gray_dir = './resize'

# 如果目标目录不存在，则创建
if not os.path.exists(gray_dir):
    os.makedirs(gray_dir)

# 遍历目录下的所有文件
for filename in os.listdir(img_dir):
    # 判断是否是图片文件
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # 构建完整的图片路径
        img_path = os.path.join(img_dir, filename)
        
        # 读取图像并转换为灰度
        img = cv2.imread(img_path)
        
        # 调整图像尺寸
        resized_img = cv2.resize(img, (640, 480))
        
        # 构建保存的路径并保存图像到gray_dir目录
        save_path = os.path.join(gray_dir, filename)
        cv2.imwrite(save_path, resized_img)