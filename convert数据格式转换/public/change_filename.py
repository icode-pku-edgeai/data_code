'''
批量文件重命名,同时修改图片和标签
'''


import os  
from pathlib import Path  
  
def rename_and_replace_jpg_files(image_path, label_path, start_number):  
    # 确保提供的文件夹路径是存在的  
    if not os.path.isdir(image_path):  
        print(f"Error: The directory {image_path} does not exist.")  
        return 
    if not os.path.isdir(label_path):  
        print(f"Error: The directory {label_path} does not exist.")  
        return  
  
    # 遍历文件夹下的所有文件  
    for filename in os.listdir(image_path):  
        # 检查文件是否是jpg文件  
        if filename.lower().endswith(('.jpg','.JPG', '.jpeg', '.png', '.gif', '.bmp', '.tiff')):  
            # 构建文件完整路径  
            old_file_path = os.path.join(image_path, filename)
            # 生成新文件名  
            new_filename = f"{start_number:06d}.jpg"  # 假设我们想要至少4位数的文件名，不足部分用0填充  
            new_file_path = os.path.join(image_path, new_filename)  
            # 增加数字以确保文件名唯一  
            while os.path.exists(new_file_path):  
                start_number += 1  
                new_filename = f"{start_number:06d}.jpg"  
                new_file_path = os.path.join(image_path, new_filename)  
  
            # 重命名文件（实际上是复制并删除原文件）  
            try:  
                Path(old_file_path).rename(new_file_path)  
                print(f"Renamed '{filename}' to '{new_filename}'")  
            except OSError as e:  
                print(f"Error: {e.strerror} when renaming {old_file_path} to {new_file_path}")  
        if filename.lower().endswith(('.jpg','.JPG', '.jpeg', '.png', '.gif', '.bmp', '.tiff')):  
                # 构建新的文件名  
            txt_name = filename.rsplit('.', 1)[0] + '.txt' 
        
        if txt_name.lower().endswith('.txt'):  
            # 构建文件完整路径  
            old_file_path = os.path.join(label_path, txt_name)
            # 生成新文件名  
            new_filename = f"{start_number:06d}.txt"  # 假设我们想要至少4位数的文件名，不足部分用0填充  
            new_file_path = os.path.join(label_path, new_filename)  
            # 增加数字以确保文件名唯一  
            while os.path.exists(new_file_path):  
                start_number += 1  
                new_filename = f"{start_number:06d}.txt"  
                new_file_path = os.path.join(label_path, new_filename)  
  
            # 重命名文件（实际上是复制并删除原文件）  
            try:  
                Path(old_file_path).rename(new_file_path)  
                print(f"Renamed '{txt_name}' to '{new_filename}'")  
            except OSError as e:  
                print(f"Error: {e.strerror} when renaming {old_file_path} to {new_file_path}") 
  
# 使用示例  
image_path = "D:\\datasets\\homemade3\\images"  # 替换为你的文件夹路径  
label_path = "D:\\datasets\\homemade3\\labels"  # 替换为你的文件夹路径  
start_number_to_use = 53639  # 从哪个数字开始编排文件名  
rename_and_replace_jpg_files(image_path, label_path,start_number_to_use)