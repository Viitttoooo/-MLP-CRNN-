# 此代码用于可视化界面中的车牌定位，相较于原版车牌定位代码有小幅修改
import cv2
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys

cur_dir = sys.path[0]


def preprocess_image(img_path,threshold_value=50, adaptive_threshold=False):
    # 读取和调整图像大小
    img = cv2.imread(img_path)
    img = cv2.resize(img, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)))

    
    lower_blue = np.array([100, 110, 110])
    upper_blue = np.array([130, 255, 255])
    # 双边滤波
    gray_filtered =cv2.GaussianBlur(img, (3, 3), 3, 0,cv2.BORDER_DEFAULT)
    median = cv2.medianBlur(gray_filtered, 5)
    hsv = cv2.cvtColor(median, cv2.COLOR_BGR2HSV)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    blue = cv2.bitwise_and(hsv, hsv, mask=mask_blue)
    # 灰度化
    gray = cv2.cvtColor(blue, cv2.COLOR_BGR2GRAY)
    # 创建并应用顶帽操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    tophat = cv2.morphologyEx(median, cv2.MORPH_TOPHAT, kernel)

    # Sobel算子应用
    sobel = cv2.Sobel(tophat, cv2.CV_16S, 1, 0)
    abs_sobel = cv2.convertScaleAbs(sobel)

    # 二值化处理
    if adaptive_threshold:
        # 使用自适应阈值
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    else:
        # 使用全局阈值
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    
    # 开运算和闭运算处理
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 8))  # 调整核的大小
    open_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))  # 调整核的大小
    close_img = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)


    # 额外的腐蚀和膨胀处理
    kernel_x = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 1))
    kernel_y = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 8))
    # erode_y = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel_y)

    # dilate_y = cv2.morphologyEx(erode_y, cv2.MORPH_DILATE, kernel_y)

    dilate_img = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel_x)
    erode_img = cv2.morphologyEx(dilate_img, cv2.MORPH_ERODE, kernel_x)

    return blue,gray, gray_filtered, tophat, abs_sobel, binary, open_img, close_img, dilate_img, erode_img


def rotate_and_crop(img, rect):
    # 获取旋转矩形的参数
    center, size, angle = rect
    center, size = tuple(map(int, center)), tuple(map(int, size))
    print("Original Center:", center)
    print("Original Size:", size)
    print("Original Angle:", angle)
    # 获取旋转矩形的变换矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1)
    
    # 旋转图像
    rotated_img = cv2.warpAffine(img, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)

    # 裁剪图像
    cropped_img = cv2.getRectSubPix(rotated_img, size, center)
    print( size[0],size[1])
    if size[0]<size[1]:
        rotated_cropped_img = cv2.rotate(cropped_img, cv2.ROTATE_90_CLOCKWISE)
    else:
        rotated_cropped_img = cropped_img
    
    
    return rotated_cropped_img

def visualize_steps(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Could not read the image.")
        exit()

    # 修改调整大小的方法
    img = cv2.resize(img, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)), interpolation=cv2.INTER_AREA)
    img_copy = img.copy()
    blue ,gray, gray_filtered, tophat, abs_sobel, binary, open_img, close_img, dilate_img, erode_img = preprocess_image(img_path)



    # 图像预处理
    processed_img = dilate_img
    

    # 画出轮廓
    contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(img_copy, contours, -1, (0, 255, 0), 3)
        
    contours_list = list(contours)

    # 计算所有轮廓的面积并进行排序
    
    sorted_contours = sorted(contours_list, key=cv2.contourArea, reverse=True)


    # 显示处理后的图像

    count = 0
    ROI = None
    
    for contour in sorted_contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
    
        # 判断宽高比例、面积，截取符合图片
        if h * 2 < w < h * 5  and count !=1:
            # 截取车牌并显示
            ROI = img[(y - 5):(y + h + 5), (x - 5):(x + w + 5)]  # 高，宽
            try:
                count += 1
                rect = cv2.minAreaRect(contour)
                rotated_cropped_img = rotate_and_crop(img, rect)
                if os.path.exists(os.path.join(cur_dir,'temp_plate/temp.jpg')): os.remove(os.path.join(cur_dir,'temp_plate/temp.jpg'))
                cv2.imwrite(os.path.join(cur_dir,'temp_plate/temp.jpg'), rotated_cropped_img)
                print("已保存车牌")
                return cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR),rotated_cropped_img, os.path.join(cur_dir,'temp_plate/temp.jpg') # 返回原图，车牌图，车牌保存路径
                # plt.show()
            except:
                print("ROI提取出错！")
                pass


