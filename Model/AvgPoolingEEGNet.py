import numpy as np
import torch
# torch.set_default_tensor_type(torch.cuda.FloatTensor)
import torch.nn as nn
from torch.nn.modules.module import _addindent


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        # print(x.shape,'55555',)  # 32 8 64 641
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        # print(super(Conv2dWithConstraint, self).forward(x).shape, '6666666666')  # 32 16 1 641
        return super(Conv2dWithConstraint, self).forward(x)


class EEGNet(nn.Module):
    def InitialBlocks(self, dropoutRate, *args, **kwargs):
        block1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.kernelLength), stride=1, padding=(0, self.kernelLength // 2), bias=False),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),

            # DepthwiseConv2D =======================
            Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.channels, 1), max_norm=1, stride=1, padding=(0, 0),
                                 groups=self.F1, bias=False),
            # ========================================

            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            # nn.ELU(),
            nn.LeakyReLU(),
            # nn.AvgPool2d((1, 4), stride=4),
            nn.AdaptiveAvgPool2d(output_size=(self.F1 * self.D, self.F1 * self.D)),
            nn.Dropout(p=dropoutRate))
        block2 = nn.Sequential(
            # SeparableConv2D =======================
            nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, self.kernelLength2), stride=1,
                      padding=(0, self.kernelLength2 // 2), bias=False, groups=self.F1 * self.D),
            nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
            # ========================================

            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
            # nn.ELU(),
            # nn.AvgPool2d((1, 8), stride=8),
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

    def __init__(self, n_classes=4, channels=60, samples=151,
                 dropoutRate=0.5, kernelLength=64, kernelLength2=16, F1=8,
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
        self.blockOutputSize_mo = self.CalculateOutSize(self.blocks_mo, channels, samples)
        self.classifierBlock_mo = self.ClassifierBlock(1024, n_classes)
        '''111'''
        # self.prob1 = nn.Conv2d(320, 10, 1)
        # self.prob2 = nn.Conv2d(128, 3, 1)
        # self.prob3 = nn.Linear(30, n_classes)

        # self.ft_prob = nn.Linear(n_classes, n_classes)
        # self.fr_prob = nn.Linear(n_classes, n_classes)

    def forward(self, x, probformer_factor=None): # x: [B T:640 N:64], probformer_factor: [B 320 128]
        '''1111'''
        # if probformer_factor:
        #     prob = probformer_factor.unsqueeze(-1)  # [B 320 128 1]
        #     prob = self.prob2(self.prob1(prob).permute(0, 2, 1, 3))  # bz 3 10 1
        #     b, p1, p2, cp = prob.shape
        #     prob = prob.reshape(b, p1 * p2 * cp)  # bz 30
        #     prob = self.prob3(prob)  # bz 10
        x = x.reshape(-1, 640, 64)
        # x = x.reshape(-1, 64, 20, 32) # bz 64 20 32
        x = x.unsqueeze(1)  # 要32 1 640 64，千万不能64*640！！！

        '''输入模型前要么640*64  要么 64*32*20/64*20*32，  千万不要64*640'''
        # print(x.shape, 'x shape before block')  # 32 1 640 64

        x = self.blocks_mo(x)
        # print(x.shape, 'x shape after blocks_mo')
        x = x.view(x.size()[0], -1)  # Flatten
        # print(x.shape, 'x shape before cb') # 32 16 8 8
        x = self.classifierBlock_mo(x) # bz 10
        '''111'''
        # if probformer_factor:
        #     g = torch.sigmoid(self.ft_prob(prob) + self.fr_prob(x))
        #     x = g * prob + (1 - g) * x

        return x

def categorical_cross_entropy(y_pred, y_true):
    # y_pred = y_pred.cuda()
    # y_true = y_true.cuda()
    y_pred = torch.clamp(y_pred, 1e-9, 1 - 1e-9)
    return -(y_true * torch.log(y_pred)).sum(dim=1).mean()

def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr