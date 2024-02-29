# 统一数据集中图片大小
import os
from PIL import Image

def resize_images(source_file, folder_path, target_size=(238, 70)):
    with open(source_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split(' ')
        image_path = parts[0]

        # 打开原始图片并调整大小
        image = Image.open(image_path)
        resized_image = image.resize(target_size, Image.LANCZOS)

        # 覆盖原始图片
        resized_image.save(image_path)

# 示例调用，注意修改路径
resize_images('final_test.txt', 'd:\final_test')