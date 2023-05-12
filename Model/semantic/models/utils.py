# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 09:47:54 2020

@author: HQ Xie
utils.py
"""
import os 
import math
import torch
import time
import torch.nn as nn
import numpy as np
# from w3lib.html import remove_tags
from nltk.translate.bleu_score import sentence_bleu
from Model.semantic.models.mutual_info import sample_batch, mutual_information
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# class BleuScore():
#     def __init__(self, w1, w2, w3, w4):
#         self.w1 = w1 # 1-gram weights
#         self.w2 = w2 # 2-grams weights
#         self.w3 = w3 # 3-grams weights
#         self.w4 = w4 # 4-grams weights
#
#     def compute_blue_score(self, real, predicted):
#         score = []
#         for (sent1, sent2) in zip(real, predicted):
#             sent1 = remove_tags(sent1).split()
#             sent2 = remove_tags(sent2).split()
#             score.append(sentence_bleu([sent1], sent2,
#                           weights=(self.w1, self.w2, self.w3, self.w4)))
#         return score
            

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        # 将数组全部填充为某一个值
        true_dist.fill_(self.smoothing / (self.size - 2)) 
        # 按照index将input重新排列 
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence) 
        # 第一行加入了<strat> 符号，不需要加入计算
        true_dist[:, self.padding_idx] = 0 #
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        self._weight_decay = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        weight_decay = self.weight_decay()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
            p['weight_decay'] = weight_decay
        self._rate = rate
        self._weight_decay = weight_decay
        # update weights
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
            
        # if step <= 3000 :
        #     lr = 1e-3
            
        # if step > 3000 and step <=9000:
        #     lr = 1e-4
             
        # if step>9000:
        #     lr = 1e-5
         
        lr = self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
  
        return lr
    

        # return lr
    
    def weight_decay(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
            
        if step <= 3000 :
            weight_decay = 1e-3
            
        if step > 3000 and step <=9000:
            weight_decay = 0.0005
             
        if step>9000:
            weight_decay = 1e-4

        weight_decay =   0
        return weight_decay

            
class SeqtoText:
    def __init__(self, vocb_dictionary, end_idx):
        self.reverse_word_map = dict(zip(vocb_dictionary.values(), vocb_dictionary.keys()))
        self.end_idx = end_idx
        
    def sequence_to_text(self, list_of_indices):
        # Looking up words in dictionary
        words = []
        for idx in list_of_indices:
            if idx == self.end_idx:
                break
            else:
                words.append(self.reverse_word_map.get(idx))
        words = ' '.join(words)
        return(words) 


class Channels():

    def AWGN(self, Tx_sig, n_var):
        Rx_sig = Tx_sig + torch.normal(0, n_var, size=Tx_sig.shape).to(device)
        return Rx_sig

    def Rayleigh(self, Tx_sig, n_var):
        shape = Tx_sig.shape
        H_real = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
        H_imag = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, n_var)
        # Channel estimation

        # print(Rx_sig, H)
        Rx_sig = torch.matmul(Rx_sig.cpu(), torch.inverse(H.cpu())).view(shape).cuda()

        return Rx_sig

    def Rician(self, Tx_sig, n_var, K=1):
        shape = Tx_sig.shape
        mean = math.sqrt(K / (K + 1))
        std = math.sqrt(1 / (K + 1))
        H_real = torch.normal(mean, std, size=[1]).to(device)
        H_imag = torch.normal(mean, std, size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, n_var)
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)

        return Rx_sig

def initNetParams(model):
    '''Init net parameters.'''
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
         
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    # 产生上三角矩阵   https://blog.csdn.net/weixin_37724529/article/details/102881776
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask)    #torch.from_numpy()用来将数组array转换为张量Tensor

    
def create_masks(src, trg, padding_idx):

    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor) #[batch, 1, seq_len]

    trg_mask = (trg == padding_idx).unsqueeze(-2).type(torch.FloatTensor) #[batch, 1, seq_len]
    look_ahead_mask = subsequent_mask(trg.size(-1)).type_as(trg_mask.data)
    combined_mask = torch.max(trg_mask, look_ahead_mask)        #???
    
    return src_mask.to(device), combined_mask.to(device)

def loss_function(x, trg, padding_idx, criterion):
    
    loss = criterion(x, trg)
    mask = (trg != padding_idx).type_as(loss.data)
    # a = mask.cpu().numpy()
    loss *= mask
    
    return loss.mean()

def PowerNormalize(x):
    
    x_square = torch.mul(x, x)      #torch.mul(a, b)是矩阵a和b对应位相乘,维度不同可以自动补足
    power = torch.mean(x_square).sqrt()     #求所有数的平均值，https://blog.csdn.net/qq_40714949/article/details/115485140
    if power > 1:
        x = torch.div(x, power)     #张量和标量做逐元素除法或者两个可广播的张量之间做逐元素除法
    
    return x


def SNR_to_noise(snr):
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)

    return noise_std

def train_step(model, data, n_var, opt, criterion, channel, mi_net=None, extraflag=False):
    model.train()
    x = data[0]
    y = data[-1]
    data_shape = data[1]
    channels = Channels()

    
    # src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)
    # print(x.is_cuda,'x train step')
    enc_output = model.encoder(x, data_shape)  # B T N -> B L
    channel_enc_output = model.channel_encoder(enc_output)  # B L -> B L'

    if extraflag:
        channel_enc_output = model.extra_add(channel_enc_output, id, t)   # id and t 怎么引入模型呢？另设数据集？
    else:
        pass

    Tx_sig = PowerNormalize(channel_enc_output)  # B L'


    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    channel_dec_output, id, t = model.channel_decoder(Rx_sig)
    dec_output = model.decoder(channel_dec_output) # B class
    pred = dec_output  # B class
    pre = F.softmax(pred, dim=1)  # 每行求和为1  到底要还是不要好呢

    #y_est = x +  torch.matmul(n, torch.inverse(H))
    #loss1 = torch.mean(torch.pow((x_est - y_est.view(x_est.shape)), 2))
    # print(pre, y)
    loss = criterion(pre, y)
    # print('pre: ', pred, 'loss: ', loss)
    if mi_net is not None:
        mi_net.eval()
        joint, marginal = sample_batch(Tx_sig, Rx_sig)
        mi_lb, _, _ = mutual_information(joint, marginal, mi_net)
        loss_mine = -mi_lb
        loss = loss + 0.0009 * loss_mine        # 公式14？

    # loss = loss_function(pred, trg_real, pad)
    opt.zero_grad()
    loss.backward(torch.ones_like(loss), retain_graph = True)
    opt.step()
    # print(loss, loss.detach().numpy().sum())

    return loss.detach().cpu().numpy().sum()


def train_mi(model, mi_net, data, n_var, opt, channel):  # done
    x = data[0]
    y = data[-1]
    data_shape = data[1]
    mi_net.train()

    channels = Channels()
    # encoder
    # src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(device)  # [batch, 1, seq_len]
    # print(x.is_cuda,'train_mi x')  # true
    enc_output = model.encoder(x, data_shape)  # B T N -> B L
    # print(enc_output.is_cuda, 'train_mi x')  #
    channel_enc_output = model.channel_encoder(enc_output)  # B L -> B L'
    Tx_sig = PowerNormalize(channel_enc_output)     # x: B L' (B 256)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)       # y = hx + n 公式（2）  n_var = 0.1
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    joint, marginal = sample_batch(Tx_sig, Rx_sig)  # B L' and B L' -> B L'
    mi_lb, _, _ = mutual_information(joint, marginal, mi_net)
    loss_mine = -mi_lb

    opt.zero_grad()
    loss_mine.backward()
    torch.nn.utils.clip_grad_norm_(mi_net.parameters(), 10.0)
    opt.step()
    # print(loss_mine, 'train_mi loss')
    return loss_mine.item()

def val_step(model, data, n_var, criterion, channel, extraflag=False):
    channels = Channels()
    x = data[0]
    y = data[-1]
    data_shape = data[1]

    # src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)

    enc_output = model.encoder(x,data_shape)
    channel_enc_output = model.channel_encoder(enc_output)
    if extraflag:
        channel_enc_output = model.extra_add(channel_enc_output, id, t)  # id and t 怎么引入模型呢？另设数据集？
    Tx_sig = PowerNormalize(channel_enc_output)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    channel_dec_output = model.channel_decoder(Rx_sig)
    dec_output = model.decoder(channel_dec_output)
    pred = dec_output

    loss = criterion(pred, y)
    pre_list = F.softmax(pred, dim=1).max(dim=1, keepdim=True)[1]

    correct = 0
    for i in range(len(y)):
        if pre_list[i] == y[i]:
            correct+=1
    # loss = loss_function(pred, trg_real, pad)
    
    return loss.item(), correct
    
def greedy_decode(model, src, n_var, max_len, padding_idx, start_symbol, channel):
    """ 
    这里采用贪婪解码器，如果需要更好的性能情况下，可以使用beam search decode
    """
    # create src_mask
    channels = Channels()
    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(device) #[batch, 1, seq_len]

    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")
            
    #channel_enc_output = model.blind_csi(channel_enc_output)
          
    memory = model.channel_decoder(Rx_sig)
    
    outputs = torch.ones(src.size(0), 1).fill_(start_symbol).type_as(src.data)

    for i in range(max_len - 1):
        # create the decode mask
        trg_mask = (outputs == padding_idx).unsqueeze(-2).type(torch.FloatTensor) #[batch, 1, seq_len]
        look_ahead_mask = subsequent_mask(outputs.size(1)).type(torch.FloatTensor)
#        print(look_ahead_mask)
        combined_mask = torch.max(trg_mask, look_ahead_mask)
        combined_mask = combined_mask.to(device)

        # decode the received signal
        dec_output = model.decoder(outputs, memory, combined_mask, None)
        pred = model.dense(dec_output)
        
        # predict the word
        prob = pred[: ,-1:, :]  # (batch_size, 1, vocab_size)
        #prob = prob.squeeze()

        # return the max-prob index
        _, next_word = torch.max(prob, dim = -1)
        #next_word = next_word.unsqueeze(1)
        
        #next_word = next_word.data[0]
        outputs = torch.cat([outputs, next_word], dim=1)

    return outputs



