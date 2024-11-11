'''
标签清洗
清洗内容：
1、标签行前后空白字符
2、清洗非YOLO格式5个数字的标签
3、超过类别范畴的标签
'''

import os  
import re  
  
def modify_files_in_directory(directory):  
    # 遍历指定目录下的所有文件  
    for filename in os.listdir(directory):  
        if filename.endswith(".txt"):  # 确保只处理txt文件  
            file_path = os.path.join(directory, filename)  
              
            # 读取文件内容  
            with open(file_path, 'r', encoding='utf-8') as file:  
                lines = file.readlines()  
              
            # 准备一个列表来存储符合条件的行  
            modified_lines = []  
              
            # 遍历每一行  
            for line in lines:  
                # 去除行首和行尾的空白字符  
                stripped_line = line.strip()  
                  
                # 尝试按空格分割字符串为浮点数列表  
                try:  
                    numbers = list(map(float, stripped_line.split()))  
                except ValueError:  
                    # 如果无法全部转换为浮点数，则跳过这行  
                    continue  
                  
                # 检查是否包含恰好5个浮点数  
                if len(numbers) != 5:  
                    continue  
                  
                # 检查第一个数字是否为0  
                if numbers[0] != 0:  
                    continue  
                  
                # 如果满足所有条件，则添加到modified_lines列表中  
                modified_lines.append(line)  
              
            # 将修改后的内容写回文件  
            with open(file_path, 'w', encoding='utf-8') as file:  
                file.writelines(modified_lines)  
  
# 指定目录路径  
directory_path = 'D:\\datasets\\homemade3\\labels'  
modify_files_in_directory(directory_path)