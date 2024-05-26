import os


def keep_lines_starting_with_char(directory, char):
    """
    保留指定目录下所有txt文件中开头为特定字符的行，删除其余行。
    
    :param directory: str, 要搜索并修改的目录路径。
    :param char: str, 要保留的行首字符。
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
            modified_lines = []  # 用于存储修改后的内容
            
            # 读取文件并筛选行
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    stripped_line = line.strip()
                    if stripped_line.startswith(char):
                        modified_lines.append(stripped_line + '\n')  # 符合条件的行加入列表并添加换行符
            
            # 将筛选后的内容写回文件
            with open(file_path, 'w', encoding='utf-8') as file:
                file.writelines(modified_lines)
            
            print(f"{filename} 文件已修改。")

# 使用示例
directory_path = 'F:/训练集图片/trainV4/VOCdevkit - 副本/labels/train'  # 指定要修改的目录路径
char_to_keep = '0'  # 指定要保留的行首字符
keep_lines_starting_with_char(directory_path, char_to_keep)