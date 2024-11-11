'''
批量进行视频resize成规定的尺寸,并拼接成一个视频
'''

import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips

def resize_and_pad(video_path, target_size=(640, 480)):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 计算缩放比例
    scale = min(target_size[0] / width, target_size[1] / height)
    new_size = (int(width * scale), int(height * scale))
    
    # 创建目标视频文件
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = video_path.replace('.mp4', '_resized.mp4')  # 输出路径
    out = cv2.VideoWriter(output_path, fourcc, fps, target_size)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 调整大小
        resized_frame = cv2.resize(frame, new_size)
        
        # 创建填充图像
        padded_frame = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        
        # 计算居中填充的位置
        pad_x = (target_size[0] - new_size[0]) // 2
        pad_y = (target_size[1] - new_size[1]) // 2
        
        # 将调整后的帧放入填充图像的中心
        padded_frame[pad_y:pad_y + new_size[1], pad_x:pad_x + new_size[0]] = resized_frame
        
        # 写入新的视频帧
        out.write(padded_frame)

    cap.release()
    out.release()
    
    return output_path

def concatenate_videos(video_folder):
    video_clips = []
    
    for filename in os.listdir(video_folder):
        if filename.endswith('.mp4'):
            video_path = os.path.join(video_folder, filename)
            resized_video_path = resize_and_pad(video_path)
            video_clips.append(VideoFileClip(resized_video_path))

    final_video = concatenate_videoclips(video_clips)
    final_video_path = os.path.join(video_folder, 'tank_multi_640_480_fps30.mp4')
    final_video.write_videofile(final_video_path)

# 使用示例
video_folder = 'D:\\datasets\\origin_video\\show_video\\tank\\result\\multi'  # 替换为你的文件夹路径
concatenate_videos(video_folder)
