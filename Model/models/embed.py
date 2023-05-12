import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random

class Embedding_1(nn.Module):
    def __init__(self, d_model, d_channel):
        super(Embedding_1, self).__init__()
        self.cov1_15 = nn.Conv1d(in_channels=15, out_channels=d_model,
                                    kernel_size=1)
        self.cov2_15 = nn.Conv1d(in_channels=4, out_channels=d_channel,
                              kernel_size=1)
    def forward(self, x1):  
        # print(x1, 'after embed')
        x1 = self.cov2_15(self.cov1_15(x1).permute(0, 2, 1)).permute(0, 2, 1)  # b*dmodel*dchannel
        # print(x1.shape, 'after embed shape')
        return x1 # b 320 128


class Embedding_2(nn.Module):
    def __init__(self, d_model, d_channel):
        super(Embedding_2, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.conv1_eye = nn.Conv1d(in_channels=1, out_channels=d_model,
                                    kernel_size=3, padding=padding, padding_mode='circular')
        self.conv2_eye = nn.Conv1d(in_channels=14, out_channels=d_channel,
                                   kernel_size=1, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x2):  
        x2 = self.conv2_eye(self.conv1_eye(x2).permute(0, 2, 1)).permute(0, 2, 1)  #  b*dmodel*dchannel  b 320 128
        return x2  # b 320 128


class Embedding_3(nn.Module):
    def __init__(self, d_model, d_channel):
        super(Embedding_3, self).__init__()
    def forward(self, x3):  # b 640 64
        x3 = x3.reshape(-1, 320, 128)  # b t n
        return x3  # b 320 128


class Embedding_4(nn.Module):
    def __init__(self, d_model, d_channel):
        super(Embedding_4, self).__init__()
        self.cov1_epoc = nn.Conv1d(in_channels=70, out_channels=d_model,
                                   kernel_size=1)
        self.cov2_epoc = nn.Conv1d(in_channels=70, out_channels=d_channel,
                                   kernel_size=1)
    def forward(self, x4):  # b 70 70
        # print(x4.shape)
        x4 = self.cov2_epoc(self.cov1_epoc(x4).permute(0, 2, 1)).permute(0, 2, 1)  # b*dmodel*dchannel
        return x4  # b 320 128

class DataEmbedding(nn.Module):
    def __init__(self, d_model=512, d_channel=96, bz=5, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.eb_15 = Embedding_1(d_model, d_channel)
        self.eb_eye = Embedding_2(d_model, d_channel)
        self.eb_mo = Embedding_3(d_model, d_channel)
        self.eb_epoc = Embedding_4(d_model, d_channel)
        self.embed_data = torch.empty(bz, d_model, d_channel)

    def forward(self, x, data_shape, data_mask=False): 
        temp = self.embed_data.clone()  
        if not data_mask:
            for bz_idx in range(len(x)):
                T, N = data_shape[bz_idx][0], data_shape[bz_idx][1]
                # print(data_shape[bz_idx], T, N)  # [15 1] 15 1
                # print('WE use embed {0} to do because N ={1}'.format(1 if N==1 else 2, N))
                # print(x[bz_idx].shape, 'before embed shape')  # d_model, d_channel
                temp[bz_idx] = self.eb1(x[bz_idx]) if N == 1 else self.eb2(x[bz_idx]) 
 
                x = x  # B L D
        else:  
            for bz_idx in range(len(x)):
                T, N = data_shape[bz_idx][0], data_shape[bz_idx][1]
                mask_tensor = torch.ones_like(x[bz_idx]).to('cuda:0')
                count = 0
                for ii, jj in enumerate(mask_tensor,0):
                    # print(mask_tensor[ii])
                    if count % 2:
                        mask_tensor[ii] = float(0)
                    count+=1

                x[bz_idx] = x[bz_idx]*mask_tensor  
            if N == 70:
                x = self.eb_epoc(x)
            elif N == 4:
                x = self.eb_15(x)
            elif N == 14:
                x = self.eb_eye(x)
            else:
                x = self.eb_mo(x)

        return self.dropout(x)  # B 320 128
