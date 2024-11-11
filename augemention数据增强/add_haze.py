'''
加雾数据增强
顶部随机雾化中心、随机雾浓度和尺寸、两图叠加
'''

import numpy as np
import random
import cv2
import math
import os
from PIL import Image
import glob

def demo(i,img_path):
    # img_path = '9060.jpg' # 图片地址和名称，默认是同一层文件地址，如有需要可更改。

    img = cv2.imread(img_path)
    img_f = img / 255.0 # 归一化
    (row, col, chs) = img.shape

    A = 0.5  # 亮度
    beta = 0.01*random.randint(3,6) # 雾的浓度
    # beta = betaa[random.randint(0,len(betaa)-1)] # 随机初始化雾的浓度
    size = 1.1*math.sqrt(max(row, col))  # 雾化尺寸，可根据自己的条件进行调节，一般的范围在中心点位置但不是很大，可自己手动设置参数
    # size = 40 # 这是我自己设置的参数，效果很不错
    center = (0, col // 2)  # 雾化中心 就是图片的中心
    for j in range(row):
        for l in range(col):
            d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
            td = math.exp(-beta * d)
            img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td) # 标准光学模型，图片的RGB三通道进行加雾
    cv2.imwrite(f'{i + 9242}.jpg', img_f*255) # 图片生成名字，切记务必要回复图片 *255，否则生成图片错误，可以尝试
    # cv2.imshow("src", img)
    # cv2.imshow("dst", img_f) # 显示图片

if __name__ == '__main__':

    png_dir = '数据地址'

    # # 修改为你想要存储jpg图片的路径
    # jpg_dir = ''
    #
    # # 保证目标文件夹存在
    # if not os.path.exists(jpg_dir):
    #     os.makedirs(jpg_dir)

    # 获取png文件列表
    png_files = [f for f in os.listdir(png_dir) if f.endswith('.jpg')]

    # 按照文件名排序
    # sorted_png_files = sorted(png_files)

    # 批量转换并重命名
    for i, png_file in enumerate(png_files):
        demo(i,png_file)





