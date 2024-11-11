'''
去除多余的图片或标签文件
'''
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import glob
import os



def search_file(data_dir, pattern=r'\.png$'):
    root_dir = os.path.abspath(data_dir)
    for root, dirs, files in os.walk(root_dir):
        for f in files:
                yield f


#去除多余图片文件
bdd_label_dir = r"/data/datasets/new_dataset/nanotrack_dataset/coco_tank_20240901/labels/val/"
# print("begin!")
for path in search_file(bdd_label_dir, r"\.txt$"):
  txtfile = path.replace("txt","jpg")
  txt_path = os.path.join(r"/data/datasets/new_dataset/nanotrack_dataset/coco_tank_20240901/images/val/", txtfile)
  jpg_path = os.path.join(bdd_label_dir, path)
  if not os.path.exists(txt_path):
    print(txtfile)
    # os.remove(jpg_path)

##去除多余标签文件
# bdd_label_dir = r"/data/datasets/new_dataset/nanotrack_dataset/coco_tank_20240901/images/val/"
# # print("begin!")
# for path in search_file(bdd_label_dir, r"\.jpg$"):
#   txtfile = path.replace("jpg","txt")
#   txt_path = os.path.join(r"/data/datasets/new_dataset/nanotrack_dataset/coco_tank_20240901/labels/val/", txtfile)
#   jpg_path = os.path.join(bdd_label_dir, path)
#   if not os.path.exists(txt_path):
#     print(txtfile)
#     # os.remove(jpg_path)
