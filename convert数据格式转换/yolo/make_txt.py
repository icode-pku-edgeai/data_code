'''
生成包含文件路径的txt文件
'''
import os

save_dir  = "/datasets/datasets/new_dataset/QAT_dataset2"
train_dir = "/datasets/datasets/new_dataset/QAT_dataset2/images/train"
train_txt_path = os.path.join(save_dir, "train2017.txt")

with open(train_txt_path, "w") as f:
    for filename in os.listdir(train_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"): # 添加你的图像文件扩展名
            file_path = os.path.join(train_dir, filename)
            f.write(file_path + "\n")

print(f"train2017.txt has been created at {train_txt_path}")

val_dir = "/datasets/datasets/new_dataset/QAT_dataset2/images/val"
val_txt_path = os.path.join(save_dir, "val2017.txt")

with open(val_txt_path, "w") as f:
    for filename in os.listdir(val_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"): # 添加你的图像文件扩展名
            file_path = os.path.join(val_dir, filename)
            f.write(file_path + "\n")

print(f"val2017.txt has been created at {val_txt_path}")

