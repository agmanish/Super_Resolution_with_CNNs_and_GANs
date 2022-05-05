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

from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor


class Generator(nn.Module):
    def __init__(self) -> None:
        super(Generator, self).__init__()
        # First conv layer.
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, (9, 9), (1, 1), (4, 4)),
            nn.PReLU(),
        )

        # Features trunk blocks.
        trunk = []
        for _ in range(16):
            trunk.append(ResidualConvBlock(64))
        self.trunk = nn.Sequential(*trunk)

        # Second conv layer.
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(64),
        )

        # Upscale block
        upsampling = []
        for _ in range(2):
            upsampling.append(UpsampleBlock(64))
        self.upsampling = nn.Sequential(*upsampling)

        # Output layer.
        self.conv_block3 = nn.Conv2d(64, 3, (9, 9), (1, 1), (4, 4))

        # Initialize neural network weights
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.conv_block1(x)
        out = self.trunk(out1)
        out2 = self.conv_block2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv_block3(out)

        out = torch.clamp_(out, 0.0, 1.0)

        return out


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            # input size. (3) x 96 x 96
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            # state size. (64) x 48 x 48
            nn.Conv2d(64, 64, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # state size. (128) x 24 x 24
            nn.Conv2d(128, 128, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # state size. (256) x 12 x 12
            nn.Conv2d(256, 256, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 6 x 6
            nn.Conv2d(512, 512, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return outoid(self.net(x).view(batch_size))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
       super(ResidualConvBlock, self).__init__()
       self.rcb = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
      identity = x
      out = self.rcb(x)
      out = torch.add(out, identity)

      return out


class UpsampleBLock(nn.Module):
     super(UpsampleBlock, self).__init__()
     self.upsample_block = nn.Sequential(
            nn.Conv2d(channels, channels * 4, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.upsample_block(x)

        return out

def psnr(outputs, labels):
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
    'PSNR': psnr,
    'SSIM': ssim,
    # could add more metrics such as accuracy for each token type
}
