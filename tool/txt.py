import os

def batch_modify_txt_files(directory, target_string, replacement_string):
    """
    批量修改指定目录下所有txt文件中的目标字符串为新字符串。
    
    :param directory: str, 要搜索并修改的目录路径。
    :param target_string: str, 需要被替换的目标字符串。
    :param replacement_string: str, 替换目标字符串的新字符串。
    """
    # 确保目录存在
    if not os.path.isdir(directory):
        print(f"目录 {directory} 不存在。")
        return
    
    # 遍历目录下的所有文件
    for filename in os.listdir(directory):
        # 只处理txt文件
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as file:
                file_data = file.read()
                
            # 替换目标字符串
            modified_data = file_data.replace(target_string, replacement_string)
            
            # 将修改后的内容写回文件
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(modified_data)
            
            print(f"{filename} 文件已修改。")

# 使用示例
directory_path = 'F:/训练集图片/trainV4/新建文件夹/label'  # 指定要修改的目录路径
old_string = '0 '  # 需要被替换的字符串
new_string = '1 '  # 新的字符串
batch_modify_txt_files(directory_path, old_string, new_string)