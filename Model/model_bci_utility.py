import torch
import torch.nn as nn
import os
import numpy as np
import math
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
# import xlrd
import math
import re
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
# import serial

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def creatTensor(data, flag):
    data = np.array(data)
    data = torch.tensor(data)
    if flag:
        data = data.float()
    else:
        data = data.long()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # print('creattensor device',device)
    data = data.to(device)
    return data  # torch.tensor


class One_hot_encoder(nn.Module):
    def __init__(self, embed_size, T = 180):
        super(One_hot_encoder, self).__init__()
        self.embed_size = embed_size
        # self.time_num = time_num
        # self.I = nn.Parameter(torch.eye(time_num, time_num, requires_grad=True)) 
        self.onehot_Linear = nn.Linear(T, 4)  
        

    def forward(self, N = 4, T=180):
        onehot = torch.eye(T, T)  # 180 180
        # print(onehot)
        onehot = self.onehot_Linear(onehot).permute(1,0)
        # print(onehot.shape)
        # print('****', onehot.shape)
        onehot = onehot.expand(self.embed_size, N, T ).permute(1, 2, 0)
        # print('****', onehot.shape)
        return onehot  # [4,180,32]

def trainModel(trainloader, model, criterion, optimizer, train_ifo):  
    total_loss = 0
    trainset, start, epoch = train_ifo[0], train_ifo[1], train_ifo[2]
    model.train()
    print('trainloader!',trainloader)
    for i, (x_data, y_data) in enumerate(trainloader, 1):  
        input = creatTensor(x_data, 1)  # torch.Size([bz, 15]) or [bz, ?, ?]bz，t，dimension
        try:
            input = input.permute(0, 2, 1)  # [bz,1,15]
        except:
            input = input.squeeze(-2)  # [bz,1,15]

        # print('input.shape',input.shape)
        target = creatTensor(y_data, 0) 
        output = model(input)  # input:tensor [1,4,180]
        loss = criterion(output, target) 
        total_loss = total_loss + loss.item()  
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step()  

        if i % 100 == 0:
            print('[{}] Epoch {} '.format(time_since(start), epoch), end='')
            print('[{}/{}] '.format(i * len(input[0]), len(trainset)), end='')
            print('loss={}'.format(loss.item()))
    print('######################Epoch {0} total_loss={1}#########################'.format(epoch, total_loss))
    return total_loss / len(trainset)  

def valModel(valloader, model, criterion, train_ifo):
    model.eval()
    total_loss = 0
    epoch = train_ifo[-1]
    correct = 0
    for i, (x_data, y_data) in enumerate(valloader, 1): 
        input = creatTensor(x_data, 1)  # torch.Size([bz, 15, 1]) or [bz, ?, ?]bz，t，dimension
        input = input.permute(0, 2, 1)  # [bz,1,15]
        # print('input.shape',input.shape)
        target = creatTensor(y_data, 0)  
        output = model(input)  # input:tensor [1,4,180]

        total = total + x_data.size(0)  
        val_result = output.max(dim=1, keepdim=True)[1]  
        correct += val_result.eq(target.view_as(val_result)).sum().item()

        loss = criterion(output, target)
        total_loss = total_loss + loss.item()

    percent = (correct * 100 / total)
    print('-------------------------Epoch {0}  valdataset total_loss：{1}, accuracy：{2} -------------'.format(epoch, total_loss, percent))
    model.train()
    return total_loss, percent


def testModel(testloader, classifier):
    correct = 0
    total = 0
    predict_list = []
    target_list = []
    y_score = []
    y_one_hot = []
    with torch.no_grad():
        for i, (x_data, y_data) in enumerate(testloader, 1):
            input = creatTensor(x_data, 1)

            input = input.permute(0, 2, 1)
            print(input.shape)
            target = creatTensor(y_data, 0)
            target_list.append(y_data)
            # print('ydata shape',y_data.shape)
            temp=[0]*4
            temp[int(y_data)] = 1
            # print('int(y_data)',int(y_data))
            y_one_hot.append(temp)
            output = classifier(input)
            y_score.append(np.array(output))
            print()
            print('*****************')
            print('output:',output)
            predict_result = output.max(dim=1, keepdim=True)[1] 
            #print(predict_result.shape)
            predict_list.append(predict_result)
            total = total + x_data.size(0)
            print(predict_result, 'pred')
            print(target, 'target')
            print('*****************')
            correct += predict_result.eq(target.view_as(predict_result)).sum().item()
            percent = '%f' % (correct * 100 / total)
            print(f'Test set: Accuracy {correct}/{total} {percent}%')
        # print('one htot',np.array(y_one_hot))
        # print('score',np.array(y_score))
        return [target_list, np.array(y_one_hot), predict_list, np.array(y_score)]




if __name__ == "__main__":
    k = torch.Tensor(1,2,3,4,5)
    q = torch.Tensor(1,2,3,4,5)
    print(k.shape,q.shape)
    kk = k.transpose(-1,-2)
    print(kk.shape)
    qk = torch.matmul(q,kk)
    print(qk.shape)
    qkv = torch.matmul(qk,q)
    print(qkv.shape)
