# 测试代码，测试模型在测试集上的准确率，注意，此代码在本地即可运行，无需在spark集群中运行
import os
import torch
from model import CRNN
from PIL import Image
from torchvision import transforms
import os
import sys
import numpy as np
from mydataset import CRNNDataSet
from torch.utils.data import DataLoader

cur_dir = sys.path[0]

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
 
# test if crnn work
 
if __name__ == '__main__':
 
	val_lines=open(os.path.join(cur_dir,'final_test.txt'),'r').readlines()
	valData = CRNNDataSet(lines=val_lines,train=False,img_width=200)
	valLoader = DataLoader(dataset=valData, batch_size=1, shuffle=False, num_workers=1)
	img_h = 32   #opt.img_h  图高度限制32，可以自行设置
	use_gpu = False # opt.use_gpu 是否使用gpu
	modelpath = os.path.join(cur_dir,'model.pth')
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

	count = 0
	cnt1 = 0
	for step, (features, label) in enumerate(valLoader, 1):
		label = label.cpu().detach().numpy()
		preds = model(features)
		preds = preds.cpu().detach().numpy()
		preds = np.argmax(preds,axis=-1)
		preds = np.transpose(preds,(1,0))
		out = []
		ll=[]
		for lab in label:
			a=lab[lab!=-1]
			b=[char_set[i] for i in a]
			b="".join(b)
			ll.append(b)
		for pred in preds:
			out_text = decode(pred,char_set)
			out.append(out_text)
		if out == ll:
			count = count + 1
		if len(out[0]) != len(ll[0]):
			cnt1 = cnt1 + 1
		if out != ll:
			if len(out[0]) != len(ll[0]):
				print('predict == >', out, 'true == >', ll, "    长度不一致")
			else:
				print('predict == >', out, 'true == >', ll)
	print('Accuracy = ', count/len(valLoader))
	print("长度不一致数量 = ", cnt1, "占比为 = ", cnt1/130)