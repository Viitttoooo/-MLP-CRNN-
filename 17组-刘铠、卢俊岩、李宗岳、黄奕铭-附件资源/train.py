# 此代码为分布式环境下的训练代码，请确保已经安装相关库
import horovod.torch as hvd
from horovod.spark import run
from pyspark.sql import SparkSession
from model import CRNN
from mydataset import CRNNDataSet
from torch.utils.data import DataLoader
import torch
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import io
from hdfs import InsecureClient

# 创建HDFS客户端
client = InsecureClient('http://192.168.0.167:50070', user='master')

spark = SparkSession.builder.appName("HorovodPyTorch").getOrCreate()

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

hvd.init()
torch.cuda.set_device(hvd.local_rank())

batch_size = 128
n_class = 66

# 请修改路径
with client.read('/crnn/final_train.txt') as train_file:
    train_lines = [line.decode('utf-8') for line in train_file.readlines()]

with client.read('/crnn/final_test.txt') as val_file:
    val_lines = [line.decode('utf-8') for line in val_file.readlines()]

trainData = CRNNDataSet(lines=train_lines,train=True,img_width=200)
train_sampler = torch.utils.data.distributed.DistributedSampler(
    trainData, num_replicas=hvd.size(), rank=hvd.rank())
trainLoader = DataLoader(
    dataset=trainData, batch_size=batch_size, shuffle=False, sampler=train_sampler, num_workers=1)

valData = CRNNDataSet(lines=val_lines,train=False,img_width=200)
val_sampler = torch.utils.data.distributed.DistributedSampler(
    valData, num_replicas=hvd.size(), rank=hvd.rank())
valLoader = DataLoader(
    dataset=valData, batch_size=batch_size, shuffle=False, sampler=val_sampler, num_workers=1)

device = torch.device('cuda:{}'.format(hvd.local_rank()) if torch.cuda.is_available() else 'cpu')
net = CRNN(imgHeight=32, nChannel=1, nClass=n_class, nHidden=256)
net = net.to(device)
loss_func = torch.nn.CTCLoss(blank=n_class - 1).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001 * hvd.size(), betas=(0.5, 0.999))
optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=net.named_parameters())
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

# Horovod: 广播参数和优化器状态
hvd.broadcast_parameters(net.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

Epoch = 50
epoch_step = len(train_lines) / batch_size
trainLoss = []
valLoss = []
trainAcc = []
valAcc = []
for epoch in range(1, Epoch + 1):
    print("进入Epoch: ", epoch)
    train_sampler.set_epoch(epoch)
    net.train()
    train_total_loss = 0
    val_total_loss = 0
    train_total_acc = 0
    val_total_acc = 0
    print("训练开始")
    for step, (features, label) in enumerate(trainLoader, 1):
        print(step)
        labels = torch.IntTensor([])
        for j in range(label.size(0)):
            labels = torch.cat((labels, label[j]), 0)
        labels=labels[labels!=-1]
        features = features.to(device)
        labels = labels.to(device)
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
    print("训练结束")
    train_total_loss = hvd.allreduce(torch.tensor(train_total_loss), name="avg_train_loss").item()
    train_total_acc = hvd.allreduce(torch.tensor(train_total_acc), name="avg_train_acc").item()
    if hvd.rank() == 0:
        print("计算损失和精确度")
        trainLoss.append(train_total_loss/step)
        trainAcc.append(train_total_acc/step)
        print("损失和精确度计算完毕")

        print("向hdfs中保存模型")
        try:
            buffer = io.BytesIO()
            print("内存块开辟完成")
            torch.save(net.state_dict(), buffer)
            print("成功写入内存")
            with client.write('/crnn/model.pth') as writer:
                print("模型开始写入hdfs")
                writer.write(buffer.getvalue())
            print("模型保存完毕")
        except Exception as e:
            print("保存模型时出错:", e)
        finally:
            buffer.close()
    
    print("进入测试模式")
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
            log_probs = out.log_softmax(2)
            targets = labels
            input_lengths = torch.IntTensor([out.size(0)] * int(out.size(1)))
            target_lengths = torch.where(label != -1, 1, 0).sum(dim=-1)
            loss = loss_func(log_probs, targets, input_lengths, target_lengths)
            acc = getAcc(out, label)
            val_total_loss+=loss
            val_total_acc+=acc
        val_total_loss = hvd.allreduce(torch.tensor(val_total_loss), name="avg_loss").item()
        val_total_acc = hvd.allreduce(torch.tensor(val_total_acc), name="avg_acc").item()
        if hvd.rank() == 0:
            valLoss.append(val_total_loss / step)
            valAcc.append(val_total_acc / step)
            print("Loss:",valLoss[-1])
            print("Acc:",valAcc[-1])
        lr_scheduler.step()

    if hvd.rank() == 0:
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

    
    spark.stop()

