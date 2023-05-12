import time
import numpy as np
import os
from Model.args import *

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class realtest_embed:
    def __init__(self, path):
        self.path = path  # './data/'
        self.data_name_list = os.listdir(self.path)
        print('Predicting: ', self.path, self.data_name_list)
        self.input_list = []
        # self.directory_name = []
        if 'brainlink' in self.path:
            data_i = os.listdir(self.path)
            # print(data_i)
            x_15 = self.embedding_15(data_i, self.data_name_list)  # 2, 15, 4
            for one_data in x_15:  # 15, 4
                self.input_list.append(one_data)
        elif 'Emotiv' in self.path:
            data_i = os.listdir(self.path)
            x_epoc = self.embedding_epoc(data_i, self.data_name_list)
            for one_data in x_epoc:  # 80, 70
                self.input_list.append(one_data[5:-5])


    def __iter__(self):

        return iter(self.input_list)

    def input_result(self):

        return iter(self.input_list)

    def embedding_15(self, data_i, data_name):
        x_embedding = []
        for j in data_i:
            x = []
            # y = []
            with open(self.path + "/" + j) as f:
                for i in f.readline().split(" "):
                    try:
                        x.append(float(i))
                    except:
                        pass
            xx = []
            for ii, jj in enumerate(x):
                kk = []
                if ii == 0:
                    kk.append(jj - 50)
                    kk.append(0.1)
                    kk.append(x[ii + 1] - jj)
                    kk.append(np.mean([x[ii + 1], jj]) - jj)
                elif ii == len(x) - 1:
                    kk.append(jj - 50)
                    kk.append(jj - x[ii - 1])
                    kk.append(0.1)
                    kk.append(np.mean([jj, x[ii - 1]]))
                else:
                    kk.append(jj - 50)
                    kk.append(jj - x[ii - 1])
                    kk.append(x[ii + 1] - jj)
                    kk.append(np.mean([x[ii - 1], jj, x[ii + 1]]))
                xx.append(kk)
            x_embedding.append(xx)
        return x_embedding

    def embedding_epoc(self, data_i, data_name):
        x_embedding = []
        for j in data_i:
            with open(self.path + "/" + j) as f:
                lines = f.readlines()
                x_temp = []
                for line in lines:
                    x = []
                    for i in line.split(','):
                        x.append(float(i))
                    x_temp.append(x)
            x_embedding.append(x_temp)
        return x_embedding



def creatTensor(data, flag='float'):
    try:
        for ind, x in enumerate(data):
            data[ind] = data[ind].detach().numpy()
    except:
        pass
    data = np.array(data)
    data = torch.tensor(data)
    if flag == 'float':
        data = data.float()
    elif flag == 'long':
        data = data.long()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    return data  # torch.tensor


def test_loader(model, input_list):
    model.eval()
    pre_list = []
    with torch.no_grad():
        for datas in input_list:
            datas = np.array(datas)
            shape = [datas.shape]
            if shape[0][-1] == 70:
                datas = creatTensor(datas, flag='float').unsqueeze(0)
                prob = model.info_enc(datas, shape)
                preds = model.AvPEEGNet_enc(datas, probformer_factor=prob)
                preds = torch.argmax(preds, dim=1)
                pre_list.append(preds)
                print('Emotiv Epoc input data\'s predict vs label: ', preds)

            if shape[0][-1] == 4:
                datas = creatTensor(datas, flag='float').unsqueeze(0)
                prob = model.info_enc(datas, shape)
                preds = model.STN_enc(datas, shape, probformer_factor=prob)
                preds = torch.argmax(preds, dim=1)
                pre_list.append(preds.to('cpu'))
                print('Brainlink input data\'s predict vs label: ', preds)
    return pre_list


def Real_Test(file_path):
    model_path = r'***\multi-brain-system\Model/realtest'  # replace the Absolute path to the folder
    model = torch.load(model_path)
    input_list = realtest_embed(file_path)
    t1 = time.time()
    pre_list = test_loader(model, input_list)
    t2 = time.time()
    print('Semantic encoding time ', t2 - t1)
    return pre_list




