# 修改图片名称
import os

def rename_images(source_file, destination_file, folder_path):
    with open(source_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    modified_lines = []
    for line in lines:
        parts = line.strip().split(' ')
        image_path = parts[0]

        image_name = os.path.basename(image_path)  # 获取图片文件名
        new_image_name = '_'.join(parts[1:]) + '.jpg'  # 构建修改后的图片文件名，使用下划线连接数字部分

        new_image_path = os.path.join(folder_path, new_image_name)  # 构建新的图片路径

        if os.path.exists(new_image_path):
            os.remove(new_image_path)  # 删除已存在的文件

        if os.path.exists(image_path):
            os.rename(image_path, new_image_path)  # 重命名图片文件

        modified_line = new_image_path + ' ' + ' '.join(parts[1:])
        modified_lines.append(modified_line + '\n')

    with open(destination_file, 'w', encoding='utf-8') as file:
        file.writelines(modified_lines)

# 示例调用，注意修改路径
rename_images('final_train.txt', 'final_train1.txt', 'd:\final_train')