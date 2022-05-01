import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_networks import *
import skimage as sk
import math

import pytorch_ssim as ps
from torch.autograd import Variable
from skimage.metrics import peak_signal_noise_ratio,structural_similarity

# class DRRN(nn.Module
class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
        self.input = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.ins = nn.InstanceNorm2d(128)
        
        nn.init.kaiming_normal_(self.input.weight)
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.output.weight)


    def forward(self, x):
        residual = x
        inputs = self.input(x)
        out = inputs
        for _ in range(9):
            tmp = out
            out = self.conv1(self.relu(out))
            out = self.ins(out)
            out =  self.conv2(self.relu(out))
            out = self.ins(out)
#             out = self.conv2(self.relu(self.conv1(self.relu(out))))
            out = torch.add(out, tmp)

        out = self.output(self.relu(out))
        out = torch.add(out, residual)
        return out

def loss_fn(outputs, labels):
    N, C, H, W = outputs.shape
        
    mse_loss = torch.sum((outputs - labels) ** 2) / N / C   # each photo, each channel
    mse_loss *= 255 * 255
    mse_loss /= H * W  
    # average loss on each pixel(0-255)
    return mse_loss

def accuracy(outputs, labels):
    N, _, _, _ = outputs.shape
    psnr = 0
    for i in range(N):
        psnr += peak_signal_noise_ratio(labels[i],outputs[i])
    return psnr / N

def ssim(outputs, labels) :
    N, _, _, _ = outputs.shape
    ssim = 0
    for i in range(N):
        
        ssim += structural_similarity(labels[i],outputs[i], win_size=3, multichannel=True)
    return ssim / N   

# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'PSNR': accuracy,
    'SSIM': ssim,
    # could add more metrics such as accuracy for each token type
}