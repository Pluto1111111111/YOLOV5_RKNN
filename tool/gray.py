from PIL import Image
import os
 
# 定义图片目录
img_dir = './img'  #原图片文件夹路径
gray_dir='./gray_img' #灰度化后保存路径

# 遍历目录下的所有文件
for filename in os.listdir(img_dir):
    # 判断是否是图片文件
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # 打开图片
        img_path = os.path.join(img_dir, filename)
        with Image.open(img_path) as img:
            # 转换为灰度图像
            # s转换为三通道灰度图 img.convert('RGB').convert('L') 
            gray_img = img.convert('L')
            # 保存灰度图像
            gray_img_path = os.path.join(gray_dir, 'gray_' + filename)
            gray_img.save(gray_img_path)