import os

def flip_yolo_boxes(input_file, output_file, image_width):
    """
    左右翻转YOLO格式的标签文件中的bounding box坐标。

    :param input_file: str, 输入的YOLO标签文件路径。
    :param output_file: str, 输出的翻转后YOLO标签文件路径。
    :param image_width: int, 图像的宽度，用于计算翻转后的坐标。
    """
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            parts = line.strip().split()  # 分割每一行
            class_index, x_center, y_center, width, height = map(float, parts)  # 转换为浮点数
            
            # 左右翻转逻辑：x_center = 1 - x_center（因为是归一化的坐标）
            flipped_x_center = 1.0 - x_center
            
            # 写入翻转后的坐标
            f_out.write(f"{int(class_index)} {flipped_x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def batch_flip_yolo_boxes(input_dir, output_dir, image_width):
    """
    批量处理目录下的所有YOLO标签文件，左右翻转其bounding box坐标。

    :param input_dir: str, 包含YOLO标签文件的输入目录路径。
    :param output_dir: str, 输出翻转后标签文件的目录路径。
    :param image_width: int, 图像的宽度，用于计算翻转后的坐标。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # 确保输出目录存在

    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):  # 确保只处理.txt文件
            input_file_path = os.path.join(input_dir, filename)
            output_file_path = os.path.join(output_dir, 'Flip_' + filename)
            flip_yolo_boxes(input_file_path, output_file_path, image_width)

# 使用示例
input_directory = './labels'  # 包含YOLO标签文件的目录
output_directory = './label_'  # 输出翻转后标签文件的目录
image_width = 1920  # 图像的宽度

batch_flip_yolo_boxes(input_directory, output_directory, image_width)