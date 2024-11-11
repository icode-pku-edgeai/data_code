'''
去掉GOT数据集txt文件中首位的内容
'''
import os
count=0
while True:
    count+=1
    image_path="C:\\Users\\li\\Desktop\\repo\\SiamTrackers-master\\NanoTrack\\data\\origin_dataset\\homemade\\train"
    filepath = f"{count:06d}"  
    filepath = os.path.join(image_path, filepath)
    file_path = os.path.join(filepath, "000001.txt")

    # 打开原始文件以读取  
    try:
        with open(file_path, 'r', encoding='utf-8') as file:  
        # 读取文件的每一行          
            lines = file.readlines()  

    except FileNotFoundError:  
        # 如果文件不存在，打印消息并继续  
        print(f"The file does not exist. Skipping...") 
        if count>100:
            break
        continue
    # 准备写入新文件的内容  
    processed_lines = []  
    
    # 对每一行进行处理  
    for line in lines:  
        # 去除每行开头的第一个数字和第一个逗号  
        # 假设每行格式都是'数字,数字,数字,...'这样的形式  
        # 使用split方法分割字符串，然后重新组合  
        parts = line.split(',')  
        # if len(parts) > 1:  # 确保至少有两个部分（以防出现单行仅有一个数字的情况）  
        processed_line = ','.join(parts[1:])  # 将除了第一个部分以外的所有部分用逗号连接  
        processed_lines.append(processed_line )  # 加上换行符以便正确写入新文件  
        # else:  
        #     # 如果行中只有一个数字（或空行），则直接写入空行或原始行（根据需要选择）  
        #     processed_lines.append('\n')  # 这里选择写入空行  
    
    # 将处理后的内容写入新文件  

    new_filepath = os.path.join(filepath, "groundtruth.txt")
    with open(new_filepath, 'w', encoding='utf-8') as file:  
        file.writelines(processed_lines)  
    os.remove(file_path)
    print("处理完成，并已保存到output.txt")