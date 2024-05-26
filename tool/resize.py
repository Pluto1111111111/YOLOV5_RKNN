import os
import cv2

# 定义图片目录
img_dir = './yuan'
resize_dir = './resize'

# 如果目标目录不存在，则创建
if not os.path.exists(resize_dir):
    os.makedirs(resize_dir)

# 遍历目录下的所有文件
for filename in os.listdir(img_dir):
    # 判断是否是图片文件
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # 构建完整的图片路径
        img_path = os.path.join(img_dir, filename)
        
        img = cv2.imread(img_path)
        # 调整图像尺寸
        resized_img = cv2.resize(img, (640, 480))
        
        # 构建保存的路径并保存图像到gray_dir目录
        save_path = os.path.join(resize_dir, filename)
        cv2.imwrite(save_path, resized_img)