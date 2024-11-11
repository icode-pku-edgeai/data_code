'''
加载视频并修改fps
'''

import cv2

def change_fps(input_video_path, output_video_path, new_fps=60):
    # 打开输入视频
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # 获取视频的基本信息
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 编码格式
    out = cv2.VideoWriter(output_video_path, fourcc, new_fps, (width, height))

    # 读取视频帧并写入输出视频
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 到达视频末尾

        # 写入当前帧到输出视频
        out.write(frame)

    # 释放资源
    cap.release()
    out.release()
    print(f"Conversion completed: {input_video_path} -> {output_video_path} at {new_fps}fps")

# 使用示例
input_video_path = 'D:\\datasets\\origin_video\\show_video\\tank\\result\\multi\\tank_multi_640_480_fps30.mp4'  # 输入视频路径
output_video_path = 'D:\\datasets\\origin_video\\show_video\\tank\\result\\multi\\tank_multi_640_480_fps60.mp4'  # 输出视频路径
change_fps(input_video_path, output_video_path, new_fps=60)
