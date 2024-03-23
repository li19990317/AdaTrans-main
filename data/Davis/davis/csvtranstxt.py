import os
import pandas as pd

# 获取文件夹路径
folder_path = r'D:\学习\论文_code\drugVQA-test - 3\davis'

# 获取文件夹中的所有文件名
file_list = os.listdir(folder_path)

# 遍历文件夹中的每个文件
for file_name in file_list:
    if file_name.endswith('.csv'):
        # 构建文件的完整路径
        file_path = os.path.join(folder_path, file_name)

        # 读取 CSV 文件
        df = pd.read_csv(file_path)

        # 构建 TXT 文件的文件名
        txt_file_name = os.path.splitext(file_name)[0] + '.txt'
        txt_file_path = os.path.join(folder_path, txt_file_name)

        # 将数据写入 TXT 文件
        df.to_csv(txt_file_path, sep='\t', index=False)  # 使用制表符作为分隔符，index=False 不写入行索引
