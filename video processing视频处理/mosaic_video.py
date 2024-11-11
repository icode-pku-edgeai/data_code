'''
视频制定区域打上马赛克
'''


import cv2
import numpy as np

def add_mosaic_to_video(input_video_path, output_video_path, regions, mosaic_size=10):
    # 打开视频文件
    cap = cv2.VideoCapture(input_video_path)
    
    # 获取视频的宽度、高度和帧率
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 创建一个 VideoWriter 对象用于输出处理后的视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4 编码
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 在每个指定区域添加马赛克
        for (x, y, width, height) in regions:
            # 确保区域在视频帧内
            if x + width <= frame_width and y + height <= frame_height:
                roi = frame[y:y+height, x:x+width]
                roi_mosaic = cv2.resize(roi, (mosaic_size, mosaic_size), interpolation=cv2.INTER_LINEAR)
                roi_mosaic = cv2.resize(roi_mosaic, (width, height), interpolation=cv2.INTER_NEAREST)
                
                # 将马赛克区域放回原图
                frame[y:y+height, x:x+width] = roi_mosaic

        # 写入处理后的帧
        out.write(frame)

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# 使用示例
input_video = 'D:\\datasets\\origin_video\\show_video\\tank\\result\\single\\tank-20.mp4'  # 输入视频文件路径
output_video = 'D:\\datasets\\origin_video\\show_video\\tank\\result\\single\\tank-20-mosaic.mp4'  # 输出视频文件路径
# 定义多个要添加马赛克的区域，格式为 (x, y, width, height)
regions = [
    (98, 567 ,174,59),   # 第一区域
    (969, 88 ,183,103),   # 第一区域
    # (1134, 601 ,141,110),   # 第一区域
    # (1500, 0 ,420,200),   # 第一区域

]

add_mosaic_to_video(input_video, output_video, regions)



# # 使用示例
# # input_video = 'input.mp4'  # 输入视频文件路径
# # output_video = 'output.mp4'  # 输出视频文件路径
# # x, y, width, height = 100, 100, 200, 200  # 要添加马赛克的区域坐标和大小
# input_video = 'D:\\datasets\\origin_video\\show_video\\tank_mosaic\\single\\1.mp4'  # 输入视频文件路径
# # input_video = 'D:\\datasets\\origin_video\\show_video\\tank_origin\\single\\3.mp4'  # 输入视频文件路径
# output_video = 'D:\\datasets\\origin_video\\show_video\\tank_mosaic\\single\\2.mp4'  # 输出视频文件路径
# x, y, width, height = 894, 14 ,580,128  # 要添加马赛克的区域坐标和大小
# add_mosaic_to_video(input_video, output_video, x, y, width, height)