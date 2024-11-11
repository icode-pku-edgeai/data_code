'''
批量从图片文件夹中加载成视频
跟踪数据集还原为视频
'''

import cv2
import os

def images_to_video(image_folder, output_video_file, fps=30):
    # 获取图片文件名并排序
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")])
    
    if not images:
        print(f"在文件夹 {image_folder} 中没有找到图片文件。")
        return

    # 读取第一张图片以获取尺寸
    first_image = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = first_image.shape

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或者使用 'XVID'
    video = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))

    # 将每张图片写入视频
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    video.release()  # 释放视频写入对象
    print(f"视频已保存为 {output_video_file}")

def process_subfolders(base_folder, output_folder):
    # 获取所有子文件夹
    subfolders = [f.path for f in os.scandir(base_folder) if f.is_dir()]
    # index=0
    
    for subfolder in subfolders:
        # index+=1
        # if(index>2):
        #     break
        output_video_file = os.path.join(output_folder, f"{os.path.basename(subfolder)}.mp4")
        images_to_video(subfolder, output_video_file, fps=30)

# 使用示例
base_folder = 'D:\\datasets\\TAO\\'  # 替换为包含子文件夹的文件夹路径
output_folder = 'D:\\datasets\\origin_video\\lasot_public_video\\'  # 替换为输出视频的文件夹路径
process_subfolders(base_folder, output_folder)
