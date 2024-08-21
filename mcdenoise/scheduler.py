# Hendrik Junkawitsch; Saarland University

# This module defines various learning rate scheduler

import torch.optim as optim
from enum import IntEnum 

class Schedule(IntEnum):
    STEP  = 0
    EXP   = 1
    CSA   = 2
    CAR   = 3
    CONST = 4

def getSchedulerById(index, optimizer, num_epochs):
    if   index == Schedule.STEP: return optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1, verbose=True)
    elif index == Schedule.EXP:  return optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1, verbose=True)
    elif index == Schedule.CSA:  return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, verbose=True)
    elif index == Schedule.CAR:  return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1.0e-9, verbose=True)
    else:                        return None