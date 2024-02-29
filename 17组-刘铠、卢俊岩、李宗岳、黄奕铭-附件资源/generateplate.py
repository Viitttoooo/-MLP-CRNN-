# 车牌生成代码，用于模拟夜间车牌
import cv2
import numpy as np
import random
import imgaug.augmenters as iaa
import os
import sys
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
from math import *

cur_dir = sys.path[0]


def tfactor(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    hsv[:,:,0] = hsv[:,:,0]*(0.8+ np.random.random()*0.2)
    hsv[:,:,1] = hsv[:,:,1]*(0.3+ np.random.random()*0.7)
    hsv[:,:,2] = hsv[:,:,2]*(0.2+ np.random.random()*0.8)

    img = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return img

def rotRandrom(img, factor, size):
    shape = size
    pts1 = np.float32([[0, 0], [0, shape[0]], [shape[1], 0], [shape[1], shape[0]]])
    pts2 = np.float32([[r(factor), r(factor)],
                        [ r(factor), shape[0] - r(factor)],
                        [shape[1] - r(factor), r(factor)],
                        [shape[1] - r(factor), shape[0] - r(factor)]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, size)
    return dst

def rot(img,angel,shape,max_angel):
    size_o = [shape[1],shape[0]]
    # print size_o
    # size = (shape[1]+ int(shape[0]*cos((float(max_angel )/180) * 3.14)),shape[0])
    # print size
    size = (shape[1]+ int(shape[0]*sin((float(max_angel )/180) * 3.14)),shape[0])
    # print size
    interval = abs( int( sin((float(angel) /180) * 3.14)* shape[0]))

    pts1 = np.float32([[0,0],[0,size_o[1]],[size_o[0],0],[size_o[0],size_o[1]]])
    if(angel>0):
        pts2 = np.float32([[interval,0],[0,size[1]  ],[size[0],0  ],[size[0]-interval,size_o[1]]])
    else:
        pts2 = np.float32([[0,0],[interval,size[1]  ],[size[0]-interval,0  ],[size[0],size_o[1]]])

    M  = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,size)

    return dst

def r(val):
    return int(np.random.random() * val)

def GetCh(f,val):
    img=Image.new("RGB",(45,70),(255,255,255))
    draw = ImageDraw.Draw(img)
    draw.text((0,3),val,(0,0,0),font=f)
    img = img.resize((23,70))
    A = np.array(img)
    return A

def GenCh1(f,val):
    img=Image.new("RGB", (23,70),(255,255,255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 2),val,(0,0,0),font=f)
    A = np.array(img)
    return A

def get_img():
    zfu = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
         "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新",
         "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
         "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z" ]

    # 背景模板
    TEMPLATE_IMAGE = os.path.join(cur_dir,"./template/template.bmp")

    # zfu=[str(i) for i in range(10)]
    end_index = 31 
    select = random.choices(zfu[:end_index],k=1) + random.choices(zfu[end_index:],k=6)
    lab=[zfu.index(i) for i in select]

    select="".join(select)
    fontC  = ImageFont.truetype(os.path.join(cur_dir, './font/platech.ttf') , 43, 0)
    fontE = ImageFont.truetype(os.path.join(cur_dir,'./font/platechar.ttf') ,60,0)
    #font=cv2.FONT_HERSHEY_COMPLEX
    img = np.array(Image.new(("RGB"),(226,70),(255,255,255)))
    bg = cv2.resize(cv2.imread(TEMPLATE_IMAGE),(226,70))

    #得到白底黑字
    offset = 2
    img[0:70,offset+8:offset+8+23] = GetCh(fontC,select[0]) 
    img[0:70,offset+8+23+6:offset+8+23+6+23] = GenCh1(fontE,select[1])
    for i in range(5):
        base = offset+8+23+6+23+17+i*23+i*6
        img[0:70, base:base+23]= GenCh1(fontE, select[i+2])
    
    #得到黑底白字
    img = cv2.bitwise_not(img)

    #字放到蓝色车牌背景中
    src = cv2.bitwise_or(img,bg)

    # 矩形变为平行四边形
    src = rot(src,r(20)-10,src.shape,10)

    # 旋转
    src = rotRandrom(src,5,(src.shape[1],src.shape[0]))

    #调灰度
    src = tfactor(src)

    #高斯模糊
    mylist = [1,3,5,7,9,11]
    width = random.choice(mylist)
    height = random.choice(mylist)
    src = cv2.GaussianBlur(src,(width,height),0)


    return src,lab

f_train=open(os.path.join(cur_dir,'com_train.txt'),'w')
f_val=open(os.path.join(cur_dir,'com_val.txt'),'w')

for i in range(20000):
    img,lab=get_img()
    lab=[str(i) for i in lab]
    lab=" ".join(lab)
    path=os.path.join(cur_dir,'com_train_data/'+str(i)+'.jpg')
    cv2.imwrite(path,img)
    f_train.write(path+' '+lab+'\n')
    print(i)
for i in range(5000):
    img,lab=get_img()
    lab=[str(i) for i in lab]
    lab=" ".join(lab)
    path=os.path.join(cur_dir,'com_val_data/'+str(i)+'.jpg')
    cv2.imwrite(path,img)
    f_val.write(path+' '+lab+'\n')
    print(i)



