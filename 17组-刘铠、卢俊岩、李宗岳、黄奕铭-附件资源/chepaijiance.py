import cv2
import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']

def preprocess_image(img_path, threshold_value=50, adaptive_threshold=False):
    # 加载图像并将其尺寸缩小为原来的一半，以提高处理效率
    img = cv2.imread(img_path)
    img = cv2.resize(img, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)))

    # 定义蓝色在HSV颜色空间中的范围，用于检测蓝色
    lower_blue = np.array([100, 110, 110])
    upper_blue = np.array([130, 255, 255])

    # 应用高斯模糊以平滑图像并减少噪声
    gray_filtered = cv2.GaussianBlur(img, (3, 3), 3, 0, cv2.BORDER_DEFAULT)

    # 使用中值滤波进一步减少噪声
    median = cv2.medianBlur(gray_filtered, 5)

    # 将图像转换为HSV颜色空间，用于基于颜色的过滤
    hsv = cv2.cvtColor(median, cv2.COLOR_BGR2HSV)

    # 创建掩码以提取图像中的蓝色区域
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    blue = cv2.bitwise_and(hsv, hsv, mask=mask_blue)

    # 转换为灰度图，用于阈值处理和边缘检测
    gray = cv2.cvtColor(blue, cv2.COLOR_BGR2GRAY)

    # 应用顶帽形态学操作以增强图像中的小元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    tophat = cv2.morphologyEx(median, cv2.MORPH_TOPHAT, kernel)

    # 使用Sobel算子检测水平边缘
    sobel = cv2.Sobel(tophat, cv2.CV_16S, 1, 0)
    abs_sobel = cv2.convertScaleAbs(sobel)

    # 对图像进行二值化处理，得到二值图像
    if adaptive_threshold:
        # 自适应阈值，适用于光照条件变化的情况
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    else:
        # 固定阈值处理
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    

    # 附加的膨胀和腐蚀步骤，以增强图像特征
    kernel_x = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 1))
    dilate_img = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel_x)


    return gray_filtered, median, blue, gray, tophat, abs_sobel, binary, dilate_img
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
    gray_filtered, median, blue, gray, tophat, abs_sobel, binary, dilate_img = preprocess_image(img_path)

    # 显示每一步的处理结果
    plt.figure(figsize=(10, 5))

    plt.subplots_adjust(wspace=0.1, hspace=0)

    plt.subplot(2, 4, 1), plt.imshow(cv2.cvtColor(gray_filtered, cv2.COLOR_BGR2RGB))
    plt.title('1. 高斯模糊'), plt.axis('off')

    plt.subplot(2, 4, 2), plt.imshow(cv2.cvtColor(median, cv2.COLOR_BGR2RGB))
    plt.title('2. 中值滤波'), plt.axis('off')

    plt.subplot(2, 4, 3), plt.imshow(cv2.cvtColor(blue, cv2.COLOR_BGR2RGB))
    plt.title('3. 提取蓝色区域'), plt.axis('off')

    plt.subplot(2, 4, 4), plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
    plt.title('4. 灰度化'), plt.axis('off')

    plt.subplot(2, 4, 5), plt.imshow(cv2.cvtColor(tophat, cv2.COLOR_BGR2RGB))
    plt.title('5. 顶帽操作'), plt.axis('off')

    plt.subplot(2, 4, 6), plt.imshow(cv2.cvtColor(abs_sobel, cv2.COLOR_BGR2RGB))
    plt.title('6. Sobel算子'), plt.axis('off')

    plt.subplot(2, 4, 7), plt.imshow(cv2.cvtColor(binary, cv2.COLOR_BGR2RGB))
    plt.title('7. 二值化'), plt.axis('off')

    plt.subplot(2, 4, 8), plt.imshow(cv2.cvtColor(dilate_img, cv2.COLOR_BGR2RGB))
    plt.title('8. 形态学运算'), plt.axis('off')

    plt.show()


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
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    axes[0].axis('off')
    axes[0].set_title('画出轮廓')

    axes[1].imshow(processed_img, cmap='gray')
    axes[1].axis('off')
    axes[1].set_title('预处理后的图片')

    plt.show()
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
                
                fig = plt.figure(figsize=(6, 3))
                plt.imshow(cv2.cvtColor(rotated_cropped_img, cv2.COLOR_BGR2RGB)), plt.axis('off'), plt.title("旋转裁剪后的最终图片")
                plt.show()
            except:
                print("ROI提取出错！")
                #return
                pass
def process_and_save_images(input_folder, output_folder,max_images=10000):
    # 遍历文件夹中的所有图片
    image_count = 0
    for file_name in sorted(os.listdir(input_folder)):
        if file_name.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_folder, file_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)), interpolation=cv2.INTER_AREA)
            img_copy = img.copy()
            # 进行图像处理和提取
            gray_filtered, median, blue, gray, tophat, abs_sobel, binary, dilate_img = preprocess_image(img_path)
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
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            count = 0
            ROI = None
            for contour in sorted_contours:
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
            
                # 判断宽高比例、面积，截取符合图片
                if h * 2 < w < h * 5 and count!=1 :
                    # 截取车牌并显示
                    ROI = img[(y - 5):(y + h + 5), (x - 5):(x + w + 5)]  # 高，宽
                    try:
                        count += 1
                        rect = cv2.minAreaRect(contour)
                        rotated_cropped_img = rotate_and_crop(img, rect)
                        output_path = os.path.join(output_folder, f'{file_name}')
                        cv2.imwrite(output_path, rotated_cropped_img)
                        print(f"Saved {output_path}")
                        image_count += 1
                        if image_count >= max_images:
                            print(f"Processed {max_images} images. Stopping.")
                            return
                    except:
                        print("ROI提取出错！")
                        #return
                        pass
if __name__ == '__main__':
    # 此处并在资料包中并没有附CCPD数据集，因为数据集有12G，过大，如有需要可自行下载，然后修改路径
    input_folder = "D:\CCPD2019\CCPD2019\ccpd_base"
    output_folder = os.path.join(input_folder, "result")
    # 调试时所用
    # img_path=r"C:\Users\Lu Junyan\Desktop\1\2.jpg"
    # visualize_steps(img_path) 
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    process_and_save_images(input_folder, output_folder)

