# 此代码用于GUI界面车牌识别，加载的模型为model.pth
import os
import torch
from model import CRNN
from PIL import Image
from torchvision import transforms
import sys
import numpy as np
from mydataset import CRNNDataSet
from torch.utils.data import DataLoader
import cv2

cur_dir = sys.path[0]
n_class = 66 

def decode(preds,char_set):
    preds=list(preds)
    pred_text = ''
    for i,j in enumerate(preds):
        if j==n_class-1:
            continue
        if i==0:
            pred_text+=char_set[j]
            continue
        if preds[i-1]!=j:
            pred_text += char_set[j] 
    return pred_text
 

class predict(object):
    def __init__(self,modelpath):
        self.cur_dir = sys.path[0]
        self.model = CRNN(32,1,66,256)
        if os.path.exists(modelpath):
            print('Load model from "%s" ...' % modelpath)
            self.model.load_state_dict(torch.load(modelpath))
            print('Done!')
        else:
            print("model path doesn't exit")
        use_gpu = False
        if torch.cuda.is_available and use_gpu:
            self.model.cuda()
    
    def read_img(self,imgpath):
        self.f = open(os.path.join(self.cur_dir,'temp.txt'),'w',encoding = 'utf-8')
        self.f.write(imgpath + ' 65 65 65 65 65 65 65')
        print("已经读取照片")
        self.f.close()
    
    def delete_temp(self):
        if os.path.exists(os.path.join(self.cur_dir,'temp.txt')) : os.remove(os.path.join(self.cur_dir,'temp.txt'))

    def predict_img(self):
        out = []
        char_set = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
         "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新",
         "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
         "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z" ] + [" "]
        self.f = open(os.path.join(self.cur_dir,'temp.txt'),'r',encoding = 'utf-8')
        val_lines=self.f.readlines()
        print(val_lines)
        valData = CRNNDataSet(lines=val_lines,train=False,img_width=200)
        valLoader = DataLoader(dataset=valData, batch_size=1, shuffle=False, num_workers=1)
        self.model.eval()
        for step, (features, label) in enumerate(valLoader, 1):
            preds = self.model(features)
            preds = preds.cpu().detach().numpy()
            preds = np.argmax(preds,axis=-1)
            preds = np.transpose(preds,(1,0))
            out = []
            for pred in preds:
                out_text = decode(pred,char_set)
                out.append(out_text)

        # 删除temp
        self.f.close()
        if os.path.exists(os.path.join(self.cur_dir,'temp.txt')) : os.remove(os.path.join(self.cur_dir,'temp.txt'))
        
        print(out)
        return out

class resizeNormalize(object):
	def __init__(self, size, interpolation=Image.BILINEAR):
		self.size = size
		self.interpolation = interpolation
		self.toTensor = transforms.ToTensor()
 
	def __call__(self, img):
		img = img.resize(self.size, self.interpolation)
		img = self.toTensor(img)
		img.sub_(0.5).div_(0.5)
		return img
  



# test if crnn work
              


if __name__ == '__main__':
 
	val_lines=open(os.path.join(cur_dir,'result.txt'),'r').readlines()
	valData = CRNNDataSet(lines=val_lines,train=False,img_width=200)
	valLoader = DataLoader(dataset=valData, batch_size=1, shuffle=False, num_workers=1)
	img_h = 32   #opt.img_h  图高度限制32，可以自行设置
	use_gpu = True  # opt.use_gpu 是否使用gpu
	modelpath = os.path.join(cur_dir,'model1.pth')
	#modelpath = '../train/models/pytorch-crnn.pth'
	# modelpath = opt.modelpath
	#char_set = open('char_std_5990.txt', 'r', encoding='utf-8').readlines()
	#char_set = ''.join([ch.strip('\n') for ch in char_set[1:]] + ['卍'])
	char_set = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
         "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新",
         "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
         "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z" ] + [" "]
    
	n_class = len(char_set)
	print(n_class)
	
	#from crnn_new import crnn
	model =  CRNN(img_h, 1, n_class, 256)
 
	if os.path.exists(modelpath):
		print('Load model from "%s" ...' % modelpath)
		model.load_state_dict(torch.load(modelpath))
		print('Done!')
 
	if torch.cuda.is_available and use_gpu:
		model.cuda()
 
 
	model.eval()

	for step, (features, label) in enumerate(valLoader, 1):
		label = label.cpu().detach().numpy()
		preds = model(features)
		preds = preds.cpu().detach().numpy()
		preds = np.argmax(preds,axis=-1)
		preds = np.transpose(preds,(1,0))
		out = []
		ll=[]
		# for lab in label:
		# 	a=lab[lab!=-1]
		# 	b=[char_set[i] for i in a]
		# 	b="".join(b)
		# 	ll.append(b)
		for pred in preds:
			out_text = decode(pred,char_set)
			out.append(out_text)
		# print('predict == >',out, 'true == >',ll)
		print('predict == >',out)
	# pred_text = decode(preds,char_set)
	# print('predict == >',pred_text)