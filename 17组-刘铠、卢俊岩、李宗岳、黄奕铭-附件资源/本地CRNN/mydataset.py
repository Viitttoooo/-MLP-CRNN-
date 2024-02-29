import torch
import cv2
from torch.utils.data import Dataset, DataLoader
import numpy as np
import imgaug.augmenters as iaa
import sys
import os

cur_dir = sys.path[0]


class CRNNDataSet(Dataset):

    def __init__(self, lines, train=True, img_width=100):
        super(CRNNDataSet, self).__init__()
        self.lines = lines
        self.train = train
        self.img_width = img_width
        self.T = img_width//4+1

    def __getitem__(self, index):
        image_path = self.lines[index].strip().split()[0]
        label = self.lines[index].strip().split()[1:]
        image = cv2.imread(image_path,0)
        # 图像预处理
        if self.train:
            image = self.get_random_data(image)
        else:
            image = cv2.resize(image,(self.img_width,32))

        # 标签格式转换为IntTensor
        label_max = np.ones(shape=(self.T),dtype=np.int32)*-1
        label = np.array([int(i) for i in label])
        label_max[0:len(label)] = label

        # 归一化
        image = (image/255.).astype('float32')
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image)
        label_max = torch.from_numpy(label_max)

        return image, label_max

    def __len__(self):
        return len(self.lines)

    def get_random_data(self, img):
        """随机增强图像"""
        seq = iaa.Sequential([
            iaa.Multiply((0.8, 1.3)),
            iaa.GaussianBlur(sigma=(0, 1.0)),
            iaa.Crop(percent=(0, 0.05)),
            iaa.Affine(
                scale=(0.95, 1.05),
                rotate=(-4, 4),
                cval=(100,250),
                mode=iaa.ia.ALL),
            iaa.Resize({"height": 32, "width": self.img_width})
        ])
        img = seq.augment(image=img)
        return img


if __name__ == '__main__':
    batch_size = 16
    lines = open(os.path.join(cur_dir, 'final_train.txt'), 'r').readlines()
    trainData = CRNNDataSet(lines=lines)
    trainLoader = DataLoader(dataset=trainData, batch_size=batch_size)
    for data, label in trainLoader:
        print(data.shape, label)
