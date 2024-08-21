# Hendrik Junkawitsch; Saarland University

# This module defines various loss functions

from pytorch_msssim import MS_SSIM, ms_ssim, SSIM, ssim
import torch.nn as nn
from enum import IntEnum 

class Loss(IntEnum):
    MAE     = 0
    MSE     = 1
    SSIM    = 2
    MSSSIM  = 3
    MIX     = 4
    MIX2    = 5

class MS_SSIM_Loss(MS_SSIM):
    def forward(self, X, Y):
        return (1-super(MS_SSIM_Loss, self).forward(X, Y))

class SSIM_Loss(SSIM):
    def forward(self, X, Y):
        return (1-super(SSIM_Loss, self).forward(X, Y))

# "Loss Functions for Image Restoration with Neural Networks", Hang Zhao et al. 2017
class MIX_Loss(nn.Module):
    def __init__(self):
        super(MIX_Loss, self).__init__()
        self.l1      = nn.L1Loss();
        self.ms_ssim = MS_SSIM_Loss(data_range=1.0, size_average=True, channel=3)
        
    def forward(self, X, Y):
        return self.l1(X, Y) * 0.16 + self.ms_ssim(X, Y) * 0.84

class MIX_2_Loss(nn.Module):
    def __init__(self):
        super(MIX_2_Loss, self).__init__()
        self.l2      = nn.MSELoss();
        self.ms_ssim = MS_SSIM_Loss(data_range=1.0, size_average=True, channel=3)
        
    def forward(self, X, Y):
        return self.l2(X, Y) * 0.16 + self.ms_ssim(X, Y) * 0.84

# Loss functions:
def getLossById(index):
    if   index == Loss.MAE:     return nn.L1Loss()
    elif index == Loss.MSE:     return nn.MSELoss()
    elif index == Loss.SSIM:    return SSIM_Loss(data_range=1.0, size_average=True, channel=3)
    elif index == Loss.MSSSIM:  return MS_SSIM_Loss(data_range=1.0, size_average=True, channel=3)
    elif index == Loss.MIX:     return MIX_Loss()
    elif index == Loss.MIX2:    return MIX_2_Loss()
    else:                       return None