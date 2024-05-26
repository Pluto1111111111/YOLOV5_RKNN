import os


def file_rename(input_dir, output_dir):
    """
    批量重命名目录下的所有文件
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # 确保输出目录存在

    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):  # 确保只处理.txt文件   可以改为其他文件
            input_file_path = os.path.join(input_dir, filename)
            output_file_path = os.path.join(output_dir, 'gray_' + filename)
            # 重命名文件
            new_name = f'Flip_' + filename
            os.rename(input_file_path, output_file_path)
            print(f'{filename} -> {new_name}')

# 使用示例
input_directory = './name'  # 原文件路径
output_directory = './new_name'  # 重命名路径

file_rename(input_directory, output_directory)