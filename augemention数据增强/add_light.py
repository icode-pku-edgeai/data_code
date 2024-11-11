'''
加高光数据增强(两套)
顶部随机高光中心、固定亮度、两图叠加
'''

import os
import time
import numpy as np
import cv2


def lighting(img, light):
    assert -100 <= light <= 100
    max_v = 4
    bright = (light / 100.0) / max_v
    mid = 1.0 + max_v * bright
    print('bright: ', bright, 'mid: ', mid)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    thresh = gray * gray
    t = np.mean(thresh)
    mask = np.where(thresh > t, 255, 0).astype(np.float32)
    brightrate = np.zeros_like(mask).astype(np.float32)
    h, w = img.shape[:2]
    # 遍历每个像素点
    for i in range(h):
        for j in range(w):
            if mask[i, j] == 255.0:
                mask[i, j] = mid
                brightrate[i, j] = bright
            else:
                mask[i, j] = (mid - 1.0) / t * thresh[i, j] + 1.0
                brightrate[i, j] = (1.0 / t * thresh[i, j]) * bright
    img = img / 255.0
    img = np.power(img, 1.0 / mask[:, :, np.newaxis]) * (1.0 / (1.0 - brightrate[:, :, np.newaxis]))
    img = np.clip(img, 0, 1.0) * 255.0
    return img.astype(np.uint8)


def lighting_fast(img, light):
    assert -100 <= light <= 100
    max_v = 4
    bright = (light / 100.0) / max_v
    mid = 1.0 + max_v * bright
    print('bright: ', bright, 'mid: ', mid)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    thresh = gray * gray
    t = np.mean(thresh)
    # 使用numpy来计算可以加速，速度远快于上面的遍历
    mask = np.where(thresh > t, 255, 0).astype(np.float32)
    brightrate = np.where(mask == 255.0, bright, (1.0 / t * thresh) * bright)
    mask = np.where(mask == 255.0, mid, (mid - 1.0) / t * thresh + 1.0)
    img = img / 255.0
    img = np.power(img, 1.0 / mask[:, :, np.newaxis]) * (1.0 / (1.0 - brightrate[:, :, np.newaxis]))
    img = np.clip(img, 0, 1.0) * 255.0
    return img.astype(np.uint8)


if __name__ == '__main__':


    png_dir = '数据地址'


    # # 修改为你想要存储jpg图片的路径
    # jpg_dir = '数据地址'
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
        # input_img = cv2.imread(str(png_files))
        print(i)
        input_img = cv2.imread(f'{png_file}')
        light = 50
        # start_time = time.time()
        # res = lighting(input_img, light)
        # print('time: {:.3f} s'.format(time.time() - start_time))
        # cv2.imwrite('302_lighting_{}.jpg'.format(light), res)
        # start_time = time.time()
        res = lighting_fast(input_img, light)
        # print('fast time: {:.3f} s'.format(time.time() - start_time))
        cv2.imwrite(f'{png_file}'.format(light), res)


