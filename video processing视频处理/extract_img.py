'''
从文件夹下的视频文件中批量提取图片帧，并命名
'''

import os  
import cv2  
  
def extract_frame_from_video(video_folder):  
    # 遍历指定文件夹下的所有文件  
    for filename in os.listdir(video_folder):  
        # 检查文件是否为MP4格式  
        if filename.endswith('.mp4'):  
            # 构建文件的完整路径  
            video_path = os.path.join(video_folder, filename)  
              
            # 打开视频文件  
            cap = cv2.VideoCapture(video_path)  
              
            # 检查视频是否成功打开  
            if not cap.isOpened():  
                print(f"Error: Could not open video {video_path}.")  
                continue  
              
            # 读取视频的第一帧（你可以根据需要读取其他帧，比如中间帧或最后一帧）  
            ret, frame = cap.read()  
              
            # 检查是否成功读取到帧  
            if not ret:  
                print(f"Error: Could not read frame from video {video_path}.")  
                cap.release()  
                continue  
              
            # 构建输出图片的路径和文件名（这里使用视频文件名，但将扩展名改为.png）  
            # output_image_path = os.path.splitext(video_path)[0] + '.jpg'  
            file_base = os.path.splitext(filename)[0]
            filename=file_base+'.jpg' 
            output_image_path = os.path.join(video_folder,filename) 
            # 保存提取的帧为图片文件  
            cv2.imwrite(output_image_path, frame)  
            # cv2.imshow('Display window', frame) 
            # cv2.waitKey(0)  
            # cv2.destroyAllWindows()
            # 释放视频捕获对象  
            cap.release()  
              
            print(f"Extracted frame from {video_path} and saved as {output_image_path}.")  
  
# 指定包含MP4文件的文件夹路径  
video_folder_path = 'D:\\datasets\\origin_video\\show_video\\tank\\result\\single\\'  
  
# 调用函数提取帧  
extract_frame_from_video(video_folder_path)