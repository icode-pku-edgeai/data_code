
import cv2
import os

xml_head = '''<annotation>
    <folder>VOC2007</folder>
    <filename>{}</filename>
    <source>
        <database>The VOC2007 Database</database>
        <annotation>PASCAL VOC2007</annotation>
        <image>flickr</image>
    </source>
    <size>
        <width>{}</width>
        <height>{}</height>
        <depth>{}</depth>
    </size>
    <segmented>0</segmented>
    '''
xml_obj = '''
    <object>        
        <name>{}</name>
        <pose>Rear</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{}</xmin>
            <ymin>{}</ymin>
            <xmax>{}</xmax>
            <ymax>{}</ymax>
        </bndbox>
    </object>
    '''
xml_end = '''
</annotation>'''

# 需要修改为你自己数据集的分类
# labels = ["red", "yellow","green","redleft","redright","greenleft","greenright","redforward",
#             "greenforward","yellowleft","yellowright","yellowforward","redturn","greenturn","yellowturn","off"]  # label for datasets
labels = ["tank"]
num_32=num_64=num_96=0
cnt = 0
# txt_to_xml("D:\\temp\\labels\\",
#            "D:\\temp\\images\\",
#            "D:\\temp\\xmls\\")
txt_path = os.path.join('D:\\datasets\\tank_debug\\labels\\train\\')  # yolo存放txt的文件目录
image_path = os.path.join('D:\\datasets\\tank_debug\\images\\train\\')  # 存放图片的文件目录
path = os.path.join('D:\\datasets\\tank_debug\\xmls\\')  # 存放生成xml的文件目录

for (root, dirname, files) in os.walk(image_path):  # 遍历图片文件夹
    for ft in files:
        # print(ft)
        ftxt = ft.replace('jpg', 'txt')  # ft是图片名字+扩展名，将jpg和txt替换
        fxml = ft.replace('jpg', 'xml')
        xml_path = path + fxml
        obj = ''

        img = cv2.imread(root + ft)
        img_h, img_w = img.shape[0], img.shape[1]
        head = xml_head.format(str(fxml), str(img_w), str(img_h), 3)

        with open(txt_path + ftxt, 'r') as f:  # 读取对应txt文件内容
            for line in f.readlines():
                yolo_datas = line.strip().split(' ')
                label = int(float(yolo_datas[0].strip()))
                center_x = round(float(str(yolo_datas[1]).strip()) * img_w)
                center_y = round(float(str(yolo_datas[2]).strip()) * img_h)
                bbox_width = round(float(str(yolo_datas[3]).strip()) * img_w)
                bbox_height = round(float(str(yolo_datas[4]).strip()) * img_h)
                # if bbox_height>100 or bbox_width>100:
                #     print(bbox_width)
                # print(bbox_height)
                bbox=bbox_width*bbox_height
                if bbox<= 1024:
                    num_32+=1
                elif bbox<=4096:
                    num_64+=1
                else:
                    num_96+=1



                xmin = str(int(center_x - bbox_width / 2))
                ymin = str(int(center_y - bbox_height / 2))
                xmax = str(int(center_x + bbox_width / 2))
                ymax = str(int(center_y + bbox_height / 2))
                # print(int(center_x + bbox_width / 2)-int(center_x - bbox_width / 2))
                # print(int(center_y + bbox_height / 2)-int(center_y - bbox_height / 2))
                obj += xml_obj.format(labels[label], xmin, ymin, xmax, ymax)
        with open(xml_path, 'w') as f_xml:
            f_xml.write(head + obj + xml_end)
        cnt += 1
print(num_32)
print(num_64)
print(num_96)
