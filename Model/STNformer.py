# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Model.args import *

class DeepConvNet(nn.Module):
    def __init__(self,nb_classes, Chans = 4, Samples = 60,
                dropoutRate = 0.5,args=None):
        super(DeepConvNet, self).__init__()
        self.dropoutRate = dropoutRate
        self.T = 120

        # Layer 1
        self.conv1 = nn.Conv2d(1, 25, (5, 1), padding=0)
        self.conv2 = nn.Conv2d(25, 25, (Chans, 1))
        self.batchnorm1 = nn.BatchNorm2d(25, eps=1e-05, momentum=0.1)

        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1), stride=(1, 1))

        # Laryer 2
        self.conv3 = nn.Conv2d(25, 50, (5, 1), padding=0)
        self.batchnorm2 = nn.BatchNorm2d(50, eps=1e-05, momentum=0.1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1), stride=(1, 1))

        # Layer 3
        self.conv4 = nn.Conv2d(50, 100, (5, 1), padding=0)
        self.batchnorm3 = nn.BatchNorm2d(100, eps=1e-05, momentum=0.1)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 1), stride=(1, 1))

        # Layer 4
        self.conv5 = nn.Conv2d(100, 200, (5, 1))
        self.batchnorm4 = nn.BatchNorm2d(200, eps=1e-05, momentum=0.1)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 1), stride=(1, 1))

        self.fc1 = nn.Linear(200 * 1 * 7, nb_classes)

    def forward(self, x):
        padding = ( 3, 3,5, 5)
        x = x.unsqueeze(1)
        x = F.pad(x, padding)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batchnorm1(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = F.dropout(x,self.dropoutRate)

        x = self.conv3(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = F.dropout(x,self.dropoutRate)

        x = self.conv4(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        x = self.pool3(x)
        x = F.dropout(x,self.dropoutRate)

        x = self.conv5(x)
        x = self.batchnorm4(x)
        x = F.elu(x)
        x = self.pool4(x)
        x = F.dropout(x,self.dropoutRate)
        # print(x.shape, '**********')
        x = x.view(-1, 200*1*7 )
        x = self.fc1(x)

        return x

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        K: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        V: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        attn_mask: [batch_size, n_heads, seq_len, seq_len] 可能没有
        '''
        B, n_heads, len1, len2, d_k = Q.shape
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)


        attn = nn.Softmax(dim=-1)(scores)  # [B,h,N,T,T]
        context = torch.matmul(attn, V)  # [B,h,N,T,T] matmul [B,h,N,T,C/2] = [B,h,N,T,C/2]
        return context  # [B,h,N,T,C/2]

class TMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(TMultiHeadAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"


        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, N, T, C]
        input_K: [batch_size, N, T, C]
        input_V: [batch_size, N, T, C]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        B, N, T, C = input_Q.shape
        Q = self.W_Q(input_Q).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # Q: [B, h, N, T, d_k]
        K = self.W_K(input_K).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # K: [B, h, N, T, d_k]
        V = self.W_V(input_V).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # V: [B, h, N, T, d_k]

        context = ScaledDotProductAttention()(Q, K, V)  # [B, h, N, T, d_k]
        context = context.permute(0, 2, 3, 1, 4)  # [B, N, T, h, d_k]
        context = context.reshape(B, N, T, self.heads * self.head_dim)  # [B, N, T, C]
        output = self.fc_out(context)  # [batch_size, N, T, C]
        return output

class TTransformer(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion, up_d, T_dim = 180):
        super(TTransformer, self).__init__()
        self.time_num = T_dim  # time_num = 180

        self.attention = TMultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.LeakyReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.norm2 = nn.LayerNorm(embed_size)

        self.dropout = nn.Dropout(dropout)


    def forward(self, value, key, query):
        B, N, T, C = query.shape  # 5 4 180 32
        attention = self.attention(query, query, query)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        result = out
        return result

class TTransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion, up_d):
        super(TTransformerBlock, self).__init__()
        self.TTransformer = TTransformer(embed_size, heads, dropout, forward_expansion, up_d)
        self.norm1 = self.norm1 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        x1 = self.dropout(self.norm1(self.TTransformer(value, key, query) + query))  # (B, N, T, C)
        return x1

### Encoder
class Encoder(nn.Module):
    def __init__(
            self,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            up_d
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.layers = nn.ModuleList(
            [
                TTransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                    up_d = up_d
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.dropout(x)
        for layer in self.layers:
            out = layer(out, out, out)
        return out

class Transformer(nn.Module):
    def __init__(
        self,
        embed_size=64,
        num_layers=3,
        heads=2,
        up_d = 1,
        forward_expansion=4,
        dropout=0.2,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            up_d
        )
        self.device = device


    def forward(self, src):
        enc_src = self.encoder(src)
        return enc_src



class BCITransformer(nn.Module):
    def __init__(
            self,
            in_channels=1,
            embed_size=64,
            num_layers=1,
            output_T_dim= 4,
            heads= 1,
            up_d = 1,
            dropout = 0.0,
            device = "cuda" if torch.cuda.is_available() else "cpu"

    ):
        super(BCITransformer, self).__init__()

        self.conv1_eye = nn.Conv2d(in_channels, embed_size, 1)
        self.conv1_15 = nn.Conv2d(in_channels, embed_size, 1)

        self.Transformer_eye = Transformer(
            embed_size,
            num_layers,
            heads,
            up_d,
            dropout = dropout,
            device = device
        )
        self.Transformer_15 = Transformer(
            embed_size,
            num_layers,
            heads,
            up_d,
            dropout=dropout,
            device=device
        )
        self.conv2_eye = nn.Conv2d(1, output_T_dim, 1)
        self.conv2_15 = nn.Conv2d(15, output_T_dim, 1)
        self.conv3_eye = nn.Conv2d(embed_size, 3, 1)
        self.conv3_15 = nn.Conv2d(embed_size, 1, 1)
        self.linear_eye = nn.Linear(output_T_dim*14*3, output_T_dim)
        self.linear_15 = nn.Linear(4*output_T_dim, output_T_dim)
        self.relu1 = nn.LeakyReLU()

        self.deepconv_eye = DeepConvNet(nb_classes=output_T_dim, Chans=4, Samples=180)
        self.ft_eye = nn.Linear(output_T_dim, output_T_dim)
        self.fr_eye = nn.Linear(output_T_dim, output_T_dim)

        self.fz_prob_eye = nn.Linear(output_T_dim, output_T_dim)

        self.prob1_15 = nn.Conv2d(args.d_model, 15, 1)
        self.prob2_15 = nn.Conv2d(args.d_channel, 4, 1)

    def forward(self, x, data_shape, probformer_factor):
        if probformer_factor is not None:
            prob = probformer_factor.unsqueeze(-1)  # [B 320 128 1]
            prob = self.prob2_15(self.prob1_15(prob).permute(0, 2, 1, 3))  # bz 3 10 1
            prob = prob.squeeze(-1)
        T, N = data_shape[0][0], data_shape[0][1]
        x = x.permute(0, 2, 1) + prob

        if N == 14:
            out_deep = self.deepconv_eye(x)
            x = x.unsqueeze(0)
            x = x.permute(1, 0, 2, 3)
            input_Transformer = self.conv1_eye(x)
        elif N == 4:
            x = x.unsqueeze(0)
            x = x.permute(1, 0, 2, 3)
            input_Transformer = self.conv1_15(x)

        input_Transformer = input_Transformer.permute(0, 2, 3, 1)

        if N == 14:
            output_Transformer = self.Transformer_eye(input_Transformer)
        elif N == 4:
            output_Transformer = self.Transformer_15(input_Transformer)


        output_Transformer = output_Transformer.permute(0, 2, 1, 3)

        if N == 14:
            out = self.relu1(self.conv2_eye(output_Transformer))
            out = out.permute(0, 3, 2, 1)
            out = self.conv3_eye(out)
            a, b, c, d = out.shape
            out = out.reshape(a, b * c * d)
            out = out.squeeze(0)  #
        elif N == 4:
            out = self.relu1(self.conv2_15(output_Transformer))
            out = out.permute(0, 3, 2, 1)
            out = self.conv3_15(out)
            out = out.permute(1, 0, 2, 3)
            a, b, c, d = out.shape
            out = out.reshape(a, b, c * d)
            out = out.squeeze(0)

        if N == 14:
            out = self.linear_eye(out)
            g = torch.sigmoid(self.ft_eye(out) + self.fz_prob_eye(prob) )
            op = g * out + (1 - g) * prob
            k = torch.sigmoid(self.ft_eye(op) + self.fr_eye(out_deep))
            result = k * op + (1 - g) * out_deep

        elif N == 4:
            result = self.linear_15(out)
        return result
