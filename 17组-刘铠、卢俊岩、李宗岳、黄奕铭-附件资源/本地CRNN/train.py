# 本地CRNN训练代码，无需任何配置
from model import CRNN
from mydataset import CRNNDataSet
from torch.utils.data import DataLoader
import torch
from torch import optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

cur_dir = sys.path[0]


def decode(preds):
    char_set = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
         "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新",
         "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
         "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z" ] + [" "]
    preds = list(preds)

    pred_text = ''
    for i, j in enumerate(preds):
        if j == n_class-1:
            continue
        if i == 0:
            pred_text += char_set[j]
            continue
        if preds[i-1] != j:
            pred_text += char_set[j]

    return pred_text


def getAcc(preds,labs):
    acc = 0
    char_set = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
         "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新",
         "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
         "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z" ] + [" "]

    labs = labs.cpu().detach().numpy()
    preds = preds.cpu().detach().numpy()
    preds = np.argmax(preds, axis=-1)
    preds = np.transpose(preds, (1, 0))

    out = []
    for pred in preds:
        out_txt = decode(pred)
        out.append(out_txt)

    ll = []
    for lab in labs:
        a = lab[lab != -1]
        b = [char_set[i] for i in a]
        b = "".join(b)
        ll.append(b)
    for a1, a2 in zip(out, ll):
        if a1 == a2:
            acc += 1
    return acc/batch_size


batch_size = 128
n_class = 66

train_lines = open(os.path.join(cur_dir,'final_train.txt'), 'r').readlines()
val_lines = open(os.path.join(cur_dir,'final_test.txt'), 'r').readlines()

trainData = CRNNDataSet(lines=train_lines,train=True,img_width=200)
trainLoader = DataLoader(dataset=trainData, batch_size=batch_size, shuffle=True, num_workers=1)
valData = CRNNDataSet(lines=val_lines,train=False,img_width=200)
valLoader = DataLoader(dataset=valData, batch_size=batch_size, shuffle=False, num_workers=1)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = CRNN(imgHeight=32, nChannel=1, nClass=n_class, nHidden=256)
net=net.to(device)


loss_func = torch.nn.CTCLoss(blank=n_class - 1)  # 注意，这里的CTCLoss中的 blank是指空白字符的位置，在这里是第65个,也即最后一个
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.5, 0.999))
# 学习率衰减
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)


# 画图列表
trainLoss = []
valLoss = []
trainAcc = []
valAcc = []


if __name__ == '__main__':

    Epoch = 30

    epoch_step = len(train_lines) / batch_size
    for epoch in range(1, Epoch + 1):

        net.train()

        train_total_loss = 0
        val_total_loss = 0
        train_total_acc = 0
        val_total_acc = 0

        with tqdm(total=epoch_step, desc=f'Epoch {epoch}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
            for step, (features, label) in enumerate(trainLoader, 1):
                labels = torch.IntTensor([])
                for j in range(label.size(0)):
                    labels = torch.cat((labels, label[j]), 0)

                labels=labels[labels!=-1]

                features = features.to(device)
                labels = labels.to(device)
                loss_func=loss_func.to(device)
                batch_size = features.size()[0]

                out = net(features)

                log_probs = out.log_softmax(2).requires_grad_()

                targets = labels
                input_lengths = torch.IntTensor([out.size(0)] * int(out.size(1)))
                target_lengths = torch.where(label != -1, 1, 0).sum(dim=-1)
                loss = loss_func(log_probs, targets, input_lengths, target_lengths)
                acc=getAcc(out,label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_total_loss += loss
                train_total_acc += acc
                pbar.set_postfix(**{'loss': train_total_loss.item() / (step),
                                    'acc': train_total_acc/ (step), })
                pbar.update(1)
        trainLoss.append(train_total_loss.item()/step)
        trainAcc.append(train_total_acc/step)

        # 保存模型
        torch.save(net.state_dict(), os.path.join(cur_dir,'model.pth'))
        # 验证
        net.eval()
        for step, (features, label) in enumerate(valLoader, 1):
            with torch.no_grad():
                labels = torch.IntTensor([])
                for j in range(label.size(0)):
                    labels = torch.cat((labels, label[j]), 0)

                labels = labels[labels != -1]

                features = features.to(device)
                labels = labels.to(device)
                loss_func = loss_func.to(device)
                batch_size = features.size()[0]

                out = net(features)

                log_probs = out.log_softmax(2).requires_grad_()

                targets = labels
                input_lengths = torch.IntTensor([out.size(0)] * int(out.size(1)))
                target_lengths = torch.where(label != -1, 1, 0).sum(dim=-1)
                loss = loss_func(log_probs, targets, input_lengths, target_lengths)
                acc = getAcc(out, label)
                val_total_loss+=loss
                val_total_acc+=acc

        valLoss.append(val_total_loss.item()/step)
        valAcc.append(val_total_acc/step)
        lr_scheduler.step()
        #print(trainLoss)
        #print(valLoss)
    """绘制loss acc曲线图"""
    plt.figure()
    plt.plot(trainLoss, 'r')
    plt.plot(valLoss, 'b')
    plt.title('Training and validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Loss", "Validation Loss"])
    plt.savefig('loss.png')

    plt.figure()
    plt.plot(trainAcc, 'r')
    plt.plot(valAcc, 'b')
    plt.title('Training and validation acc')
    plt.xlabel("Epochs")
    plt.ylabel("Acc")
    plt.legend(["Acc", "Validation Acc"])
    plt.savefig('acc.png')
    plt.show()
