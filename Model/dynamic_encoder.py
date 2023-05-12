import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.utils.masking import TriangularCausalMask, ProbMask
from Model.models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from Model.models.decoder import Decoder, DecoderLayer
from Model.models.attn import FullAttention, ProbAttention, AttentionLayer
from Model.models.embed import DataEmbedding
from Model.STNformer import BCITransformer as STN
from Model.AvgPoolingEEGNet import EEGNet
from Model.args import *



class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)

class embed_epoc(nn.Module):
    def __init__(self, input_size=1, output_size=64):
        super(embed_epoc, self).__init__()
        self.l = nn.Linear(input_size, output_size)

        self.l2 = nn.Linear(70, 40)
    def forward(self,x):
        x = self.l(x.unsqueeze(-1)).permute(0,3,1,2)
        x = self.l2(x)
        return x  # 32 64 15 40

class EEGNet(nn.Module):
    def InitialBlocks(self, dropoutRate, *args, **kwargs):
        block1 = nn.Sequential(
            nn.Conv2d(64, self.F1, (1, self.kernelLength), stride=1, padding=(0, self.kernelLength // 2), bias=False),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),

            Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.channels, 1), max_norm=1, stride=1, padding=(0, 0),
                                 groups=self.F1, bias=False),
            # ========================================

            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d(output_size=(self.F1 * self.D, self.F1 * self.D)),
            nn.Dropout(p=dropoutRate))
        block2 = nn.Sequential(
            # SeparableConv2D =======================
            nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, self.kernelLength2), stride=1,
                      padding=(0, self.kernelLength2 // 2), bias=False, groups=self.F1 * self.D),
            nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
            # ========================================

            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d(output_size=(self.F1, self.F1)),
            nn.Dropout(p=dropoutRate))
        return nn.Sequential(block1, block2)


    def ClassifierBlock(self, inputSize, n_classes):
        return nn.Sequential(
            nn.Linear(inputSize, n_classes, bias=False),
            nn.Softmax(dim=1))

    def CalculateOutSize(self, model, channels, samples):
        '''
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        '''
        data = torch.rand(1, 1, channels, samples)
        model.eval()
        out = model(data).shape
        return out[2:]

    def __init__(self, n_classes=10, channels=60, samples=151,
                 dropoutRate=0.5, kernelLength=12, kernelLength2=16, F1=8,
                 D=2, F2=16):
        super(EEGNet, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.samples = samples
        self.n_classes = n_classes
        self.channels = channels
        self.kernelLength = kernelLength
        self.kernelLength2 = kernelLength2
        self.dropoutRate = dropoutRate

        self.blocks_mo = self.InitialBlocks(dropoutRate)
        self.classifierBlock_mo = self.ClassifierBlock(1024, n_classes)
        self.embed_epoc = embed_epoc(input_size=1, output_size=64)
        self.prob1 = nn.Conv2d(args.d_model, 70, 1)
        self.prob2 = nn.Conv2d(args.d_channel, 64, 1)
        self.prob3 = nn.Linear(1, 40)


    def forward(self, x, probformer_factor=None):
        if probformer_factor is not None:
            prob = probformer_factor.unsqueeze(-1)  # [B 320 128 1]
            prob = self.prob2(self.prob1(prob).permute(0, 2, 1, 3))  # bz 3 10 1
            prob = self.prob3(prob)  # bz 64 70 40
        else:
            prob = 0
        x = self.embed_epoc(x)
        x = self.blocks_mo(x+prob)
        x = x.view(x.size()[0], -1)  # Flatten
        x = self.classifierBlock_mo(x)  # bz 10
        return x

class Probformer(nn.Module):
    def __init__(self,class_num=10,factor=5, d_model=128, d_channel=128, n_heads=4, e_layers=3, d_layers=2, d_ff=512,
                dropout=0.3, batchsize=50, attn='prob', activation='relu',
                output_attention = False, distil=True, mix=True,
                device= 'cpu'):

        super(Probformer, self).__init__()
        self.class_num = class_num

        # Embedding
        self.data_embedding = DataEmbedding(d_model, d_channel, batchsize, dropout=0.0)

        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_channel, n_heads, mix=False),  # BLD -> BLD
                    d_channel,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],  # in: B L D  out: B L D
            [
                ConvLayer(
                    d_channel
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_channel)
        )

        self.Linear_eye = nn.Linear(322, 320)

    def forward(self, x, data_shape):
        data_in_mask = self.data_embedding(x, data_shape, data_mask=True) # input: B T N  output: B dmodel dchannel
        enc_out, attns = self.encoder(data_in_mask, attn_mask=None)  # enc_out: B L D : 32 320 d_channel
        if data_shape[0][-1] ==14:
            enc_out = self.Linear_eye(enc_out.permute(0, 2, 1)).permute(0, 2, 1)

        return enc_out





class Senmatic_encoder(nn.Module):
    def __init__(self, output_dim, embed_size, batch_size):
        super(Senmatic_encoder, self).__init__()

        self.info_enc = Probformer(class_num=output_dim, factor=args.factor, d_model=args.d_model, d_channel=args.d_channel,
                         n_heads=args.n_heads, e_layers=args.e_layers, d_layers=args.d_layers,
                         d_ff=args.d_ff ,batchsize=batch_size, dropout=args.dropout)

        self.STN_enc = STN(in_channels=args.in_channels, embed_size=embed_size, num_layers=args.num_layers, output_T_dim= output_dim, heads= args.STN_heads, up_d=args.up_d, dropout =args.STNdropout,
                           device=args.device)

        self.AvPEEGNet_enc = EEGNet(n_classes=output_dim, channels=args.EEG_channels, samples=args.samples,
               dropoutRate=args.dropoutRate, kernelLength=args.kernelLength, kernelLength2=args.kernelLength2, F1=args.F1,
               D=args.D, F2=args.F2)
