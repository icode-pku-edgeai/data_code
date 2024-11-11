'''
图像尺寸调整,等比缩放,无padding操作
'''
from PIL import Image
from PIL.Image import Resampling 
#将图片尺寸统一  宽度是480
import os
# path = "E://数据集/原图/番茄"#读取的文件夹
# savepath = "E://数据集/原图/bb/"#保存的文件夹
path = ""#读取的文件夹
savepath = ""#保存的文件夹
filelist = os.listdir(path) #该文件夹下所有的文件（包括文件夹）
count=0
for file in filelist:
    print(file)
for file in filelist:   #遍历所有文件
    Olddir=os.path.join(path,file)   #原来的文件路径
    if os.path.isdir(Olddir):   #如果是文件夹则跳过
        continue
    filename=os.path.splitext(file)[0]   #文件名
    filetype=os.path.splitext(file)[1]   #文件扩展名
    fileall = Olddir+filename+filetype
    #print(Olddir)
    # print(fileall)
    im = Image.open(Olddir)
 
    (x, y) = im.size  # read image size
    x_s = 640  # define standard width
    y_s = y * x_s // x  # calc height based on standard width
    im=im.convert('RGB')
    out = im.resize((x_s, y_s), Resampling.LANCZOS)  # resize image with high-quality
    savepaths = savepath+filename+filetype
    #print(savepaths)
    out.save(savepaths)