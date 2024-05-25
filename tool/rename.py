import os


def file_rename(input_dir, output_dir):
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
            output_file_path = os.path.join(output_dir, 'gray_' + filename)
            # 重命名文件
            new_name = f'Flip_' + filename
            os.rename(input_file_path, output_file_path)
            print(f'{filename} -> {new_name}')

# 使用示例
input_directory = './name'  # 包含YOLO标签文件的目录
output_directory = './new_name'  # 输出翻转后标签文件的目录

file_rename(input_directory, output_directory)