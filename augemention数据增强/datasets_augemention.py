

#离线数据增强
#https://blog.csdn.net/qq_44421796/article/details/135180578?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522EA9CFFE5-BE7D-4FDF-89AD-BE9F1CCD4F10%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=EA9CFFE5-BE7D-4FDF-89AD-BE9F1CCD4F10&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-3-135180578-null-null.142^v100^control&utm_term=yolo%E6%95%B0%E6%8D%AE%E9%9B%86%E6%95%B0%E6%8D%AE%E5%A2%9E%E5%BC%BA&spm=1018.2226.3001.4187
import time
import random
import copy
import cv2
import os
import math
import numpy as np
from skimage.util import random_noise
from lxml import etree, objectify
import xml.etree.ElementTree as ET
import argparse


# 显示图片
def show_pic(img, bboxes=None):
    '''
    输入:
        img:图像array
        bboxes:图像的所有boudning box list, 格式为[[x_min, y_min, x_max, y_max]....]
        names:每个box对应的名称
    '''
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3]
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 3)
    cv2.namedWindow('pic', 0)  # 1表示原图
    cv2.moveWindow('pic', 0, 0)
    cv2.resizeWindow('pic', 1200, 800)  # 可视化的图片大小
    cv2.imshow('pic', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 图像均为cv2读取
class DataAugmentForObjectDetection():
    def __init__(self, rotation_rate=0.5, max_rotation_angle=5,
                 crop_rate=0.5, shift_rate=0.5, change_light_rate=0.5,
                 add_noise_rate=0.5, flip_rate=0.5,
                 cutout_rate=0.5, cut_out_length=50, cut_out_holes=1, cut_out_threshold=0.5,
                 is_addNoise=True, is_changeLight=True, is_cutout=True, is_rotate_img_bbox=True,
                 is_crop_img_bboxes=True, is_shift_pic_bboxes=True, is_filp_pic_bboxes=True):

        # 配置各个操作的属性
        self.rotation_rate = rotation_rate
        self.max_rotation_angle = max_rotation_angle
        self.crop_rate = crop_rate
        self.shift_rate = shift_rate
        self.change_light_rate = change_light_rate
        self.add_noise_rate = add_noise_rate
        self.flip_rate = flip_rate
        self.cutout_rate = cutout_rate

        self.cut_out_length = cut_out_length
        self.cut_out_holes = cut_out_holes
        self.cut_out_threshold = cut_out_threshold

        # 是否使用某种增强方式
        self.is_addNoise = is_addNoise
        self.is_changeLight = is_changeLight
        self.is_cutout = is_cutout
        self.is_rotate_img_bbox = is_rotate_img_bbox
        self.is_crop_img_bboxes = is_crop_img_bboxes
        self.is_shift_pic_bboxes = is_shift_pic_bboxes
        self.is_filp_pic_bboxes = is_filp_pic_bboxes

    # ----1.加噪声---- #
    def _addNoise(self, img):
        '''
        输入:
            img:图像array
        输出:
            加噪声后的图像array,由于输出的像素是在[0,1]之间,所以得乘以255
        '''
        # return cv2.GaussianBlur(img, (11, 11), 0)
        return random_noise(img, mode='gaussian', clip=True) * 255

    # ---2.调整亮度--- #
    def _changeLight(self, img):
        alpha = random.uniform(0.35, 1)
        blank = np.zeros(img.shape, img.dtype)
        return cv2.addWeighted(img, alpha, blank, 1 - alpha, 0)

    # ---3.cutout--- #
    def _cutout(self, img, bboxes, length=100, n_holes=1, threshold=0.5):
        '''
        原版本：https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
        Randomly mask out one or more patches from an image.
        Args:
            img : a 3D numpy array,(h,w,c)
            bboxes : 框的坐标
            n_holes (int): Number of patches to cut out of each image.
            length (int): The length (in pixels) of each square patch.
        '''

        def cal_iou(boxA, boxB):
            '''
            boxA, boxB为两个框，返回iou
            boxB为bouding box
            '''
            # determine the (x, y)-coordinates of the intersection rectangle
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])

            if xB <= xA or yB <= yA:
                return 0.0

            # compute the area of intersection rectangle
            interArea = (xB - xA + 1) * (yB - yA + 1)

            # compute the area of both the prediction and ground-truth
            # rectangles
            boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
            boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
            iou = interArea / float(boxBArea)
            return iou

        # 得到h和w
        if img.ndim == 3:
            h, w, c = img.shape
        else:
            _, h, w, c = img.shape
        mask = np.ones((h, w, c), np.float32)
        for n in range(n_holes):
            chongdie = True  # 看切割的区域是否与box重叠太多
            while chongdie:
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = np.clip(y - length // 2, 0,
                             h)  # numpy.clip(a, a_min, a_max, out=None), clip这个函数将将数组中的元素限制在a_min, a_max之间，大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min
                y2 = np.clip(y + length // 2, 0, h)
                x1 = np.clip(x - length // 2, 0, w)
                x2 = np.clip(x + length // 2, 0, w)

                chongdie = False
                for box in bboxes:
                    if cal_iou([x1, y1, x2, y2], box) > threshold:
                        chongdie = True
                        break
            mask[y1: y2, x1: x2, :] = 0.
        img = img * mask
        return img

    # ---4.旋转--- #
    def _rotate_img_bbox(self, img, bboxes, angle=5, scale=1.):
        '''
        参考:https://blog.csdn.net/u014540717/article/details/53301195crop_rate
        输入:
            img:图像array,(h,w,c)
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
            angle:旋转角度
            scale:默认1
        输出:
            rot_img:旋转后的图像array
            rot_bboxes:旋转后的boundingbox坐标list
        '''
        # 旋转图像
        w = img.shape[1]
        h = img.shape[0]
        # 角度变弧度
        rangle = np.deg2rad(angle)  # angle in radians
        # now calculate new image width and height
        nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
        nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
        # ask OpenCV for the rotation matrix
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
        # calculate the move from the old center to the new center combined
        # with the rotation
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        # the move only affects the translation, so update the translation
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        # 仿射变换
        rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

        # 矫正bbox坐标
        # rot_mat是最终的旋转矩阵
        # 获取原始bbox的四个中点，然后将这四个点转换到旋转后的坐标系下
        rot_bboxes = list()
        for bbox in bboxes:
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]
            point1 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymin, 1]))
            point2 = np.dot(rot_mat, np.array([xmax, (ymin + ymax) / 2, 1]))
            point3 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymax, 1]))
            point4 = np.dot(rot_mat, np.array([xmin, (ymin + ymax) / 2, 1]))
            # 合并np.array
            concat = np.vstack((point1, point2, point3, point4))
            # 改变array类型
            concat = concat.astype(np.int32)
            # 得到旋转后的坐标
            rx, ry, rw, rh = cv2.boundingRect(concat)
            rx_min = rx
            ry_min = ry
            rx_max = rx + rw
            ry_max = ry + rh
            # 加入list中
            rot_bboxes.append([rx_min, ry_min, rx_max, ry_max])

        return rot_img, rot_bboxes

    # ---5.裁剪--- #
    def _crop_img_bboxes(self, img, bboxes):
        '''
        裁剪后的图片要包含所有的框
        输入:
            img:图像array
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
        输出:
            crop_img:裁剪后的图像array
            crop_bboxes:裁剪后的bounding box的坐标list
        '''
        # 裁剪图像
        w = img.shape[1]
        h = img.shape[0]
        x_min = w  # 裁剪后的包含所有目标框的最小的框
        x_max = 0
        y_min = h
        y_max = 0
        for bbox in bboxes:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(y_max, bbox[3])

        d_to_left = x_min  # 包含所有目标框的最小框到左边的距离
        d_to_right = w - x_max  # 包含所有目标框的最小框到右边的距离
        d_to_top = y_min  # 包含所有目标框的最小框到顶端的距离
        d_to_bottom = h - y_max  # 包含所有目标框的最小框到底部的距离

        # 随机扩展这个最小框
        crop_x_min = int(x_min - random.uniform(0, d_to_left))
        crop_y_min = int(y_min - random.uniform(0, d_to_top))
        crop_x_max = int(x_max + random.uniform(0, d_to_right))
        crop_y_max = int(y_max + random.uniform(0, d_to_bottom))

        # 随机扩展这个最小框 , 防止别裁的太小
        # crop_x_min = int(x_min - random.uniform(d_to_left//2, d_to_left))
        # crop_y_min = int(y_min - random.uniform(d_to_top//2, d_to_top))
        # crop_x_max = int(x_max + random.uniform(d_to_right//2, d_to_right))
        # crop_y_max = int(y_max + random.uniform(d_to_bottom//2, d_to_bottom))

        # 确保不要越界
        crop_x_min = max(0, crop_x_min)
        crop_y_min = max(0, crop_y_min)
        crop_x_max = min(w, crop_x_max)
        crop_y_max = min(h, crop_y_max)

        crop_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

        # 裁剪boundingbox
        # 裁剪后的boundingbox坐标计算
        crop_bboxes = list()
        for bbox in bboxes:
            crop_bboxes.append([bbox[0] - crop_x_min, bbox[1] - crop_y_min, bbox[2] - crop_x_min, bbox[3] - crop_y_min])

        return crop_img, crop_bboxes

    # ---6.平移--- #
    def _shift_pic_bboxes(self, img, bboxes):
        '''
        平移后的图片要包含所有的框
        输入:
            img:图像array
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
        输出:
            shift_img:平移后的图像array
            shift_bboxes:平移后的bounding box的坐标list
        '''
        # 平移图像
        w = img.shape[1]
        h = img.shape[0]
        x_min = w  # 裁剪后的包含所有目标框的最小的框
        x_max = 0
        y_min = h
        y_max = 0
        for bbox in bboxes:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(y_max, bbox[3])

        d_to_left = x_min  # 包含所有目标框的最大左移动距离
        d_to_right = w - x_max  # 包含所有目标框的最大右移动距离
        d_to_top = y_min  # 包含所有目标框的最大上移动距离
        d_to_bottom = h - y_max  # 包含所有目标框的最大下移动距离

        x = random.uniform(-(d_to_left - 1) / 3, (d_to_right - 1) / 3)
        y = random.uniform(-(d_to_top - 1) / 3, (d_to_bottom - 1) / 3)

        M = np.float32([[1, 0, x], [0, 1, y]])  # x为向左或右移动的像素值,正为向右负为向左; y为向上或者向下移动的像素值,正为向下负为向上
        shift_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        #  平移boundingbox
        shift_bboxes = list()
        for bbox in bboxes:
            shift_bboxes.append([bbox[0] + x, bbox[1] + y, bbox[2] + x, bbox[3] + y])

        return shift_img, shift_bboxes

    # ---7.镜像--- #
    def _filp_pic_bboxes(self, img, bboxes):
        '''
            平移后的图片要包含所有的框
            输入:
                img:图像array
                bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
            输出:
                flip_img:平移后的图像array
                flip_bboxes:平移后的bounding box的坐标list
        '''
        # 翻转图像

        flip_img = copy.deepcopy(img)
        h, w, _ = img.shape

        sed = random.random()

        if 0 < sed < 0.33:  # 0.33的概率水平翻转，0.33的概率垂直翻转,0.33是对角反转
            flip_img = cv2.flip(flip_img, 0)  # _flip_x
            inver = 0
        elif 0.33 < sed < 0.66:
            flip_img = cv2.flip(flip_img, 1)  # _flip_y
            inver = 1
        else:
            flip_img = cv2.flip(flip_img, -1)  # flip_x_y
            inver = -1

        # 调整boundingbox
        flip_bboxes = list()
        for box in bboxes:
            x_min = box[0]
            y_min = box[1]
            x_max = box[2]
            y_max = box[3]

            if inver == 0:
                # 0：垂直翻转
                flip_bboxes.append([x_min, h - y_max, x_max, h - y_min])
            elif inver == 1:
                # 1：水平翻转
                flip_bboxes.append([w - x_max, y_min, w - x_min, y_max])
            elif inver == -1:
                # -1：水平垂直翻转
                flip_bboxes.append([w - x_max, h - y_max, w - x_min, h - y_min])
        return flip_img, flip_bboxes

    # 图像增强方法
    def dataAugment(self, img, bboxes):
        '''
        图像增强
        输入:
            img:图像array
            bboxes:该图像的所有框坐标
        输出:
            img:增强后的图像
            bboxes:增强后图片对应的box
        '''
        change_num = 0  # 改变的次数
        # print('------')
        while change_num < 1:  # 默认至少有一种数据增强生效

            if self.is_rotate_img_bbox:
                if random.random() > self.rotation_rate:  # 旋转
                    change_num += 1
                    angle = random.uniform(-self.max_rotation_angle, self.max_rotation_angle)
                    scale = random.uniform(0.7, 0.8)
                    img, bboxes = self._rotate_img_bbox(img, bboxes, angle, scale)

            if self.is_shift_pic_bboxes:
                if random.random() < self.shift_rate:  # 平移
                    change_num += 1
                    img, bboxes = self._shift_pic_bboxes(img, bboxes)

            if self.is_changeLight:
                if random.random() > self.change_light_rate:  # 改变亮度
                    change_num += 1
                    img = self._changeLight(img)

            if self.is_addNoise:
                if random.random() < self.add_noise_rate:  # 加噪声
                    change_num += 1
                    img = self._addNoise(img)
            if self.is_cutout:
                if random.random() < self.cutout_rate:  # cutout
                    change_num += 1
                    img = self._cutout(img, bboxes, length=self.cut_out_length, n_holes=self.cut_out_holes,
                                       threshold=self.cut_out_threshold)
            if self.is_filp_pic_bboxes:
                if random.random() < self.flip_rate:  # 翻转
                    change_num += 1
                    img, bboxes = self._filp_pic_bboxes(img, bboxes)

        return img, bboxes


# xml解析工具
class ToolHelper():
    # 从xml文件中提取bounding box信息, 格式为[[x_min, y_min, x_max, y_max, name]]
    def parse_xml(self, path):
        '''
        输入：
            xml_path: xml的文件路径
        输出：
            从xml文件中提取bounding box信息, 格式为[[x_min, y_min, x_max, y_max, name]]
        '''
        tree = ET.parse(path)
        root = tree.getroot()
        objs = root.findall('object')
        coords = list()
        for ix, obj in enumerate(objs):
            name = obj.find('name').text
            box = obj.find('bndbox')
            x_min = int(box[0].text)
            y_min = int(box[1].text)
            x_max = int(box[2].text)
            y_max = int(box[3].text)
            coords.append([x_min, y_min, x_max, y_max, name])
        return coords

    # 保存图片结果
    def save_img(self, file_name, save_folder, img):
        cv2.imwrite(os.path.join(save_folder, file_name), img)

    # 保持xml结果
    def save_xml(self, file_name, save_folder, img_info, height, width, channel, bboxs_info):
        '''
        :param file_name:文件名
        :param save_folder:#保存的xml文件的结果
        :param height:图片的信息
        :param width:图片的宽度
        :param channel:通道
        :return:
        '''
        folder_name, img_name = img_info  # 得到图片的信息

        E = objectify.ElementMaker(annotate=False)

        anno_tree = E.annotation(
            E.folder(folder_name),
            E.filename(img_name),
            E.path(os.path.join(folder_name, img_name)),
            E.source(
                E.database('Unknown'),
            ),
            E.size(
                E.width(width),
                E.height(height),
                E.depth(channel)
            ),
            E.segmented(0),
        )

        labels, bboxs = bboxs_info  # 得到边框和标签信息
        for label, box in zip(labels, bboxs):
            anno_tree.append(
                E.object(
                    E.name(label),
                    E.pose('Unspecified'),
                    E.truncated('0'),
                    E.difficult('0'),
                    E.bndbox(
                        E.xmin(box[0]),
                        E.ymin(box[1]),
                        E.xmax(box[2]),
                        E.ymax(box[3])
                    )
                ))

        etree.ElementTree(anno_tree).write(os.path.join(save_folder, file_name), pretty_print=True)


if __name__ == '__main__':

    need_aug_num = 2  # 每张图片需要增强的次数

    is_endwidth_dot = True  # 文件是否以.jpg或者png结尾

    dataAug = DataAugmentForObjectDetection()  # 数据增强工具类

    toolhelper = ToolHelper()  # 工具

    # 获取相关参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_img_path', type=str, default='D:\\datasets\\tank_debug\\images\\train')
    parser.add_argument('--source_xml_path', type=str, default='D:\\datasets\\tank_debug\\xmls')
    parser.add_argument('--save_img_path', type=str, default='D:\\datasets\\tank_debug\\test\\images')
    parser.add_argument('--save_xml_path', type=str, default='D:\\datasets\\tank_debug\\test\\xmls')
    args = parser.parse_args()
    source_img_path = args.source_img_path  # 图片原始位置
    source_xml_path = args.source_xml_path  # xml的原始位置

    save_img_path = args.save_img_path  # 图片增强结果保存文件
    save_xml_path = args.save_xml_path  # xml增强结果保存文件

    # 如果保存文件夹不存在就创建
    if not os.path.exists(save_img_path):
        os.mkdir(save_img_path)

    if not os.path.exists(save_xml_path):
        os.mkdir(save_xml_path)

    for parent, _, files in os.walk(source_img_path):
        files.sort()
        for file in files:
            cnt = 0
            pic_path = os.path.join(parent, file)
            xml_path = os.path.join(source_xml_path, file[:-4] + '.xml')
            values = toolhelper.parse_xml(xml_path)  # 解析得到box信息，格式为[[x_min,y_min,x_max,y_max,name]]
            coords = [v[:4] for v in values]  # 得到框
            labels = [v[-1] for v in values]  # 对象的标签

            # 如果图片是有后缀的
            if is_endwidth_dot:
                # 找到文件的最后名字
                dot_index = file.rfind('.')
                _file_prefix = file[:dot_index]  # 文件名的前缀
                _file_suffix = file[dot_index:]  # 文件名的后缀
            img = cv2.imread(pic_path)

            # show_pic(img, coords)  # 显示原图
            while cnt < need_aug_num:  # 继续增强
                auged_img, auged_bboxes = dataAug.dataAugment(img, coords)
                auged_bboxes_int = np.array(auged_bboxes).astype(np.int32)
                height, width, channel = auged_img.shape  # 得到图片的属性
                img_name = '{}_{}{}'.format(_file_prefix, cnt + 1, _file_suffix)  # 图片保存的信息
                toolhelper.save_img(img_name, save_img_path,
                                    auged_img)  # 保存增强图片

                toolhelper.save_xml('{}_{}.xml'.format(_file_prefix, cnt + 1),
                                    save_xml_path, (save_img_path, img_name), height, width, channel,
                                    (labels, auged_bboxes_int))  # 保存xml文件
                # show_pic(auged_img, auged_bboxes)  # 强化后的图
                print(img_name)
                cnt += 1  # 继续增强下一张










#随机copy-paste数据增强,没有标签处理
#https://blog.csdn.net/qq_37346140/article/details/130387601?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522EA9CFFE5-BE7D-4FDF-89AD-BE9F1CCD4F10%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=EA9CFFE5-BE7D-4FDF-89AD-BE9F1CCD4F10&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-130387601-null-null.142^v100^control&utm_term=yolo%E6%95%B0%E6%8D%AE%E9%9B%86%E6%95%B0%E6%8D%AE%E5%A2%9E%E5%BC%BA&spm=1018.2226.3001.4187
# import numpy as np
# import cv2
# import os
# import tqdm
# import argparse
# from skimage.draw import polygon
# import random


# def random_flip_horizontal(img, box, p=0.5):
#     '''
#     对img和mask随机进行水平翻转。box为二维np.array。
#     https://blog.csdn.net/weixin_41735859/article/details/106468551
#     img[:,:,::-1] gbr-->bgr、img[:,::-1,:] 水平翻转、img[::-1,:,:] 上下翻转
#     '''
#     if np.random.random() < p:
#         w = img.shape[1]

#         img = img[:, ::-1, :]
#         box[:, [0, 2, 4, 6]] = w - box[:, [2, 0, 6, 4]]  # 仅针对4个点变换
#     return img, box


# def Large_Scale_Jittering(img, box, min_scale=0.1, max_scale=2.0):
#     '''
#     对img和box进行0.1-2.0的大尺度抖动，并变回h*w的大小。
#     '''
#     rescale_ratio = np.random.uniform(min_scale, max_scale)
#     h, w, _ = img.shape

#     # rescale
#     h_new, w_new = int(h * rescale_ratio), int(w * rescale_ratio)
#     img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_LINEAR)

#     # crop or padding
#     # x,y是随机选择左上角的一个点，让小图片在这个位置，或者让大图片从这个位置开始裁剪
#     x, y = int(np.random.uniform(0, abs(w_new - w))), int(np.random.uniform(0, abs(h_new - h)))
#     # 如果图像缩小了，那么其余部分要填充为像素168大小
#     if rescale_ratio <= 1.0:  # padding
#         img_pad = np.ones((h, w, 3), dtype=np.uint8) * 168
#         img_pad[y:y + h_new, x:x + w_new, :] = img
#         box[:, [0, 2, 4, 6]] = box[:, [0, 2, 4, 6]] * w_new / w + x  # x坐标
#         box[:, [0, 2, 4, 6]] = box[:, [0, 2, 4, 6]] * w_new / w + x
#         box[:, [1, 3, 5, 7]] = box[:, [1, 3, 5, 7]] * h_new / h + y  # y坐标
#         return img_pad, box
#     # 如果图像放大了，那么要裁剪成h*w的大小
#     else:  # crop
#         img_crop = img[y:y + h, x:x + w, :]

#         box[:, [0, 2, 4, 6]] = box[:, [0, 2, 4, 6]] * w_new / w - x
#         box[:, [1, 3, 5, 7]] = box[:, [1, 3, 5, 7]] * h_new / h - y

#         return img_crop, box


# def img_add(img_src, img_main, mask_src, box_src):
#     '''
#     将src加到main图像中，结果图还是main图像的大小。
#     '''
#     if len(img_main.shape) == 3:
#         h, w, c = img_main.shape
#     elif len(img_main.shape) == 2:
#         h, w = img_main.shape
#     src_h, src_w = img_src.shape[0], img_src.shape[1]

#     mask = np.asarray(mask_src, dtype=np.uint8)
#     # mask是二值图片，对src进行局部遮挡，即只露出目标物体的像素。
#     sub_img01 = cv2.add(img_src, np.zeros(np.shape(img_src), dtype=np.uint8), mask=mask)  # 报错深度不一致

#     mask_02 = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
#     mask_02 = np.asarray(mask_02, dtype=np.uint8)
#     sub_img02 = cv2.add(img_main, np.zeros(np.shape(img_main), dtype=np.uint8),
#                         mask=mask_02)  # 在main图像上对应位置挖了一块

#     # main图像减去要粘贴的部分的图，然后加上复制过来的图
#     img_main = img_main - sub_img02 + cv2.resize(sub_img01, (w, h),
#                                                  interpolation=cv2.INTER_NEAREST)

#     box_src[:, [0, 2, 4, 6]] = box_src[:, [0, 2, 4, 6]] * w / src_w
#     box_src[:, [1, 3, 5, 7]] = box_src[:, [1, 3, 5, 7]] * h / src_h

#     return img_main, box_src


# def normal_(jpg_path, txt_path="", box=None):
#     """
#     根据txt获得box或者根据box获得mask。
#     :param jpg_path: 图片路径
#     :param txt_path: x1,y1,x2,y2 x3,y3,x4,y4...
#     :param box: 如果有box，则为根据box生成mask
#     :return: 图像,box 或 掩码
#     """
#     if isinstance(jpg_path, str):  # 如果是路径就读取图片
#         jpg_path = cv2.imread(jpg_path)
#     print(jpg_path)
#     # jpg_path = cv2.imread(jpg_path)
#     img = jpg_path.copy()

#     if box is None:  # 一定有txt_path
#         lines = open(txt_path).readlines()

#         box = []
#         for line in lines:
#             ceils = line.strip("\n").split(' ')
#             xy = []
#             for ceil in ceils:
#                 xy.append(round(float(ceil)))

#             x = xy[1]
#             y = xy[2]
#             w = xy[3] / 2
#             h = xy[4] / 2
#             xy = [x - w, y + h, x + w, y + h, x - w, y - h, x + w, y - h]
#             box.append(np.array(xy))

#         return np.array(img), np.array(box)

#     else:  # 获得mask
#         h, w = img.shape[:2]
#         mask = np.zeros((h, w), dtype=np.float32)

#         for xy in box:  # 对每个框
#             xy = np.array(xy).reshape(-1, 2)
#             cv2.fillPoly(mask, [xy.astype(np.int32)], 1)

#         return np.array(mask)


# def is_coincide(polygon_1, polygon_2):
#     '''
#     判断2个四边形是否重合
#     :param polygon_1: [x1, y1,...,x4, y4]
#     :param polygon_2:
#     :return:  bool，1表示重合
#     '''

#     rr1, cc1 = polygon([polygon_1[i] for i in range(0, len(polygon_1), 2)],
#                        [polygon_1[i] for i in range(1, len(polygon_1), 2)])
#     rr2, cc2 = polygon([polygon_2[i] for i in range(0, len(polygon_2), 2)],
#                        [polygon_2[i] for i in range(1, len(polygon_2), 2)])

#     try:  # 能包含2个四边形的最小矩形长宽
#         r_max = max(rr1.max(), rr2.max()) + 1
#         c_max = max(cc1.max(), cc2.max()) + 1
#     except:
#         return 0

#     # 相当于canvas是包含了2个多边形的一个画布，有2个多边形的位置像素为1，重合位置像素为2
#     canvas = np.zeros((r_max, c_max))
#     canvas[rr1, cc1] += 1
#     canvas[rr2, cc2] += 1

#     intersection = np.sum(canvas == 2)
#     return 1 if intersection != 0 else 0


# def copy_paste(img_main_path, img_src_path, txt_main_path, txt_src_path, coincide=False, muti_obj=True):
#     '''
#     整个复制粘贴操作，输入2张图的图片和坐标路径，返回其融合后的图像和坐标结果。
#     1. 传入随机选择的main图像和src图像的img和txt路径；
#     2. 对其进行随机水平翻转；
#     3. 对其进行随机抖动；
#     4. 获得src变换完后对应的mask；
#     5. 将src的结果加到main中，返回对应main_new的img和src图的box.
#     '''
#     # 读取图像和坐标
#     img_main, box_main = normal_(img_main_path, txt_main_path)
#     img_src, box_src = normal_(img_src_path, txt_src_path)

#     # 随机水平翻转
#     img_main, box_main = random_flip_horizontal(img_main, box_main)
#     img_src, box_src = random_flip_horizontal(img_src, box_src)

#     # LSJ， Large_Scale_Jittering 大尺度抖动，并变回h*w大小
#     img_main, box_main = Large_Scale_Jittering(img_main, box_main)
#     img_src, box_src = Large_Scale_Jittering(img_src, box_src)

#     if not muti_obj or box_src.ndim == 1:  # 只复制粘贴一个目标
#         id = random.randint(0, len(box_src) - 1)
#         box_src = box_src[id]
#         box_src = box_src[np.newaxis, :]  # 增加一维

#     # 获得一系列变换后的img_src的mask
#     mask_src = normal_(img_src_path, box=box_src)

#     # 将src结果加到main图像中，返回main图像的大小的叠加图
#     img, box_src = img_add(img_src, img_main, mask_src, box_src)

#     # 判断融合后的区域是否重合
#     if not coincide:
#         for point_main in box_main:
#             for point_src in box_src:
#                 if is_coincide(point_main, point_src):
#                     return None, None

#     box = np.vstack((box_main, box_src))
#     return img, box


# def save_res(img, img_path, box, txt_path):
#     '''
#     保存图片和txt坐标结果。
#     '''
#     cv2.imwrite(img_path, img)

#     h, w = img.shape[:2]
#     with open(txt_path, 'w+') as ftxt:
#         for point in box:  # [x1,y1,...x4,,y4]
#             strxy = ""
#             for i, p in enumerate(point):
#                 if i % 2 == 0:  # x坐标
#                     p = np.clip(p, 0, w - 1)
#                 else:  # y坐标
#                     p = np.clip(p, 0, h - 1)
#                 strxy = strxy + str(p) + ','
#             strxy = strxy[:-1]  # 去掉最后一个逗号
#             ftxt.writelines(strxy + "\n")


# def main(args):
#     # 图像和坐标txt文件输入路径

#     JPEGs = os.path.join(args.input_dir, 'images')
#     BOXes = os.path.join(args.input_dir, 'labels')

#     # 输出路径
#     os.makedirs(args.output_dir, exist_ok=True)
#     os.makedirs(os.path.join(args.output_dir, 'cpAug_jpg'), exist_ok=True)
#     os.makedirs(os.path.join(args.output_dir, 'cpAug_txt'), exist_ok=True)

#     # 参与数据增强的图片名称，不含后缀
#     imgs_list = open(args.aug_txt, 'r').read().splitlines()
#     flag = '.jpg'  # 图像的后缀名 .jpg ,png

#     tbar = tqdm.tqdm(imgs_list, ncols=100)  # 进度条显示
#     for src_name in tbar:
#         # src图像
#         img_src_path = os.path.join(JPEGs, src_name + flag)
#         txt_src_path = os.path.join(BOXes, src_name + '.txt')

#         # 随机选择main图像
#         main_name = np.random.choice(imgs_list)
#         img_main_path = os.path.join(JPEGs, main_name + flag)
#         txt_main_path = os.path.join(BOXes, main_name + '.txt')

#         # 数据增强
#         print(img_main_path)
#         print(img_src_path)
#         img, box = copy_paste(img_main_path, img_src_path, txt_main_path, txt_src_path,
#                               args.coincide, args.muti_obj)
#         if img is None:
#             continue

#         # 保存结果
#         img_name = "copy_" + src_name + "_paste_" + main_name
#         print(os.path.join(args.output_dir, 'cpAug_jpg', img_name + flag))
#         save_res(img, os.path.join(args.output_dir, 'cpAug_jpg', img_name + flag),
#                  box, os.path.join(args.output_dir, 'cpAug_txt', img_name + '.txt'))


# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input_dir", default="D:\\datasets\\tank_debug\\images\\train", type=str,
#                         help="要进行数据增强的图像路径，路径结构下应有jpg和txt文件夹")
#     parser.add_argument("--output_dir", default="D:/dataTest/newImage", type=str,
#                         help="保存数据增强结果的路径")
#     parser.add_argument("--aug_txt", default="D:/dataTest/oImage/test.txt",
#                         type=str, help="要进行数据增强的图像的名字，不包含后缀")
#     parser.add_argument("--coincide", default=False, type=bool,
#                         help="True表示允许数据增强后的图像目标出现重合，默认不允许重合")
#     parser.add_argument("--muti_obj", default=False, type=bool,
#                         help="True表示将src图上的所有目标都复制粘贴，False表示只随机粘贴一个目标")
#     return parser.parse_args()


# if __name__ == "__main__":
#     args = get_args()
#     main(args)





















# # # '''
# # # Author: CodingWZP
# # # Email: codingwzp@gmail.com
# # # Date: 2021-08-06 10:51:35
# # # LastEditTime: 2021-08-09 10:53:43
# # # Description: Image augmentation with label.
# # # '''
# import xml.etree.ElementTree as ET
# import os
# import imgaug as ia
# import numpy as np
# import shutil
# from tqdm import tqdm
# from PIL import Image
# from imgaug import augmenters as iaa

# ia.seed(1)


# def read_xml_annotation(root, image_id):
#     in_file = open(os.path.join(root, image_id))
#     tree = ET.parse(in_file)
#     root = tree.getroot()
#     bndboxlist = []

#     for object in root.findall('object'):  # 找到root节点下的所有country节点
#         bndbox = object.find('bndbox')  # 子节点下节点rank的值

#         xmin = int(bndbox.find('xmin').text)
#         xmax = int(bndbox.find('xmax').text)
#         ymin = int(bndbox.find('ymin').text)
#         ymax = int(bndbox.find('ymax').text)
#         # print(xmin,ymin,xmax,ymax)
#         bndboxlist.append([xmin, ymin, xmax, ymax])
#         # print(bndboxlist)

#     bndbox = root.find('object').find('bndbox')
#     return bndboxlist


# def change_xml_list_annotation(root, image_id, new_target, saveroot, id):
#     in_file = open(os.path.join(root, str(image_id) + '.xml'))  # 这里root分别由两个意思
#     tree = ET.parse(in_file)
#     # 修改增强后的xml文件中的filename
#     elem = tree.find('filename')
#     elem.text = (str(id) + '.jpg')
#     xmlroot = tree.getroot()
#     # 修改增强后的xml文件中的path
#     elem = tree.find('path')
#     if elem != None:
#         elem.text = (saveroot + str(id) + '.jpg')

#     index = 0
#     for object in xmlroot.findall('object'):  # 找到root节点下的所有country节点
#         bndbox = object.find('bndbox')  # 子节点下节点rank的值

#         # xmin = int(bndbox.find('xmin').text)
#         # xmax = int(bndbox.find('xmax').text)
#         # ymin = int(bndbox.find('ymin').text)
#         # ymax = int(bndbox.find('ymax').text)

#         new_xmin = new_target[index][0]
#         new_ymin = new_target[index][1]
#         new_xmax = new_target[index][2]
#         new_ymax = new_target[index][3]

#         xmin = bndbox.find('xmin')
#         xmin.text = str(new_xmin)
#         ymin = bndbox.find('ymin')
#         ymin.text = str(new_ymin)
#         xmax = bndbox.find('xmax')
#         xmax.text = str(new_xmax)
#         ymax = bndbox.find('ymax')
#         ymax.text = str(new_ymax)

#         index = index + 1

#     tree.write(os.path.join(saveroot, str(id + '.xml')))


# def mkdir(path):
#     # 去除首位空格
#     path = path.strip()
#     # 去除尾部 \ 符号
#     path = path.rstrip("\\")
#     # 判断路径是否存在
#     # 存在     True
#     # 不存在   False
#     isExists = os.path.exists(path)
#     # 判断结果
#     if not isExists:
#         # 如果不存在则创建目录
#         # 创建目录操作函数
#         os.makedirs(path)
#         print(path + ' 创建成功')
#         return True
#     else:
#         # 如果目录存在则不创建，并提示目录已存在
#         print(path + ' 目录已存在')
#         return False
# #     imgs_path = 'D:\\temp\\images'
# #     xmls_path = 'D:\\temp\\xmls'
# #     save_xml = 'D:\\temp\\new_xmls'
# #     save_img = 'D:\\temp\\new_images'

# if __name__ == "__main__":

#     IMG_DIR = "D:\\datasets\\tank_debug\\images\\train"
#     XML_DIR = "D:\\datasets\\tank_debug\\xmls"

#     AUG_XML_DIR = "D:\\datasets\\tank_debug\\test\\xmls"  # 存储增强后的XML文件夹路径
#     try:
#         shutil.rmtree(AUG_XML_DIR)
#     except FileNotFoundError as e:
#         a = 1
#     mkdir(AUG_XML_DIR)

#     AUG_IMG_DIR = "D:\\datasets\\tank_debug\\test\\images"  # 存储增强后的影像文件夹路径
#     try:
#         shutil.rmtree(AUG_IMG_DIR)
#     except FileNotFoundError as e:
#         a = 1
#     mkdir(AUG_IMG_DIR)

#     AUGLOOP = 2  # 每张影像增强的数量

#     boxes_img_aug_list = []
#     new_bndbox = []
#     new_bndbox_list = []

#     # 影像增强
#     seq = iaa.Sequential([
#         iaa.Invert(0.5),
#         iaa.Fliplr(0.5),  # 镜像
#         iaa.Crop(percent=(0, 0.1)),
#         iaa.Multiply((0.5, 1.5)),  # change brightness, doesn't affect BBs
#         iaa.GaussianBlur(sigma=(0, 3.0)),  # iaa.GaussianBlur(0.5),
#         iaa.Affine(
#             translate_px={"x": 15, "y": 15},
#             scale=(0.8, 0.95),
#         )  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
#     ])

#     for name in tqdm(os.listdir(XML_DIR), desc='Processing'):

#         bndbox = read_xml_annotation(XML_DIR, name)

#         # 保存原xml文件
#         # shutil.copy(os.path.join(XML_DIR, name), AUG_XML_DIR)
#         # 保存原图
#         # og_img = Image.open(IMG_DIR + '/' + name[:-4] + '.jpg')
#         # og_img.convert('RGB').save(AUG_IMG_DIR+ '/' + name[:-4] + '.jpg', 'JPEG')
#         og_xml = open(os.path.join(XML_DIR, name))
#         tree = ET.parse(og_xml)
#         # 修改增强后的xml文件中的filename
#         elem = tree.find('filename')
#         elem.text = (name[:-4] + '.jpg')
#         # tree.write(os.path.join(AUG_XML_DIR, name))

#         for epoch in range(AUGLOOP):
#             seq_det = seq.to_deterministic()  # 保持坐标和图像同步改变，而不是随机
#             # 读取图片
#             img = Image.open(os.path.join(IMG_DIR, name[:-4] + '.jpg'))
#             # sp = img.size
#             img = np.asarray(img)
#             # bndbox 坐标增强
#             for i in range(len(bndbox)):
#                 bbs = ia.BoundingBoxesOnImage([
#                     ia.BoundingBox(x1=bndbox[i][0], y1=bndbox[i][1], x2=bndbox[i][2], y2=bndbox[i][3]),
#                 ], shape=img.shape)

#                 bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
#                 boxes_img_aug_list.append(bbs_aug)

#                 # new_bndbox_list:[[x1,y1,x2,y2],...[],[]]
#                 n_x1 = int(max(1, min(img.shape[1], bbs_aug.bounding_boxes[0].x1)))
#                 n_y1 = int(max(1, min(img.shape[0], bbs_aug.bounding_boxes[0].y1)))
#                 n_x2 = int(max(1, min(img.shape[1], bbs_aug.bounding_boxes[0].x2)))
#                 n_y2 = int(max(1, min(img.shape[0], bbs_aug.bounding_boxes[0].y2)))
#                 if n_x1 == 1 and n_x1 == n_x2:
#                     n_x2 += 1
#                 if n_y1 == 1 and n_y2 == n_y1:
#                     n_y2 += 1
#                 if n_x1 >= n_x2 or n_y1 >= n_y2:
#                     print('error', name)
#                 new_bndbox_list.append([n_x1, n_y1, n_x2, n_y2])
#             # 存储变化后的图片
#             image_aug = seq_det.augment_images([img])[0]
#             path = os.path.join(AUG_IMG_DIR,
#                                 str(str(name[:-4]) + '_' + str(epoch)) + '.jpg')
#             image_auged = bbs.draw_on_image(image_aug, size=0)
#             Image.fromarray(image_auged).convert('RGB').save(path)

#             # 存储变化后的XML
#             change_xml_list_annotation(XML_DIR, name[:-4], new_bndbox_list, AUG_XML_DIR,
#                                        str(name[:-4]) + '_' + str(epoch))
#             # print(str(str(name[:-4]) + '_' + str(epoch)) + '.jpg')
#             new_bndbox_list = []
#     print('Finish!')

# -*- coding=utf-8 -*-
