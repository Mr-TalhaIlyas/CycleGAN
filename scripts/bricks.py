import yaml
with open('config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)

import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv_IN_Act(nn.Module):
    def __init__(self, inChannel, outChannel, kernel_size=4, stride=2, padding=1,
                 normalize=True, use_act=True):
        super(Conv_IN_Act, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inChannel, outChannel, kernel_size, stride=stride, padding=padding,
                      bias=True, padding_mode='reflect'),
            nn.InstanceNorm2d(outChannel) if normalize else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True) if use_act else nn.Identity()
            )

    def forward(self, x):
        return self.conv(x)

class TransConvBlock(nn.Module):
    def __init__(self, inChannel, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(inChannel, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, inChannel):
        super().__init__()
        self.block = nn.Sequential(
            Conv_IN_Act(inChannel, inChannel, kernel_size=3, stride=1, padding=1),
            Conv_IN_Act(inChannel, inChannel, use_act=False, kernel_size=3, stride=1, padding=1)
        )
    
    def forward(self, x):
        return x + self.block(x)