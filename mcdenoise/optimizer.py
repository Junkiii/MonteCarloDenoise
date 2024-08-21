# Hendrik Junkawitsch; Saarland University

# This module defines the optimization algorithms

from enum import IntEnum
import torch.optim as optim
from torch.optim.sgd import SGD

class Optimizer(IntEnum):
    ADAM = 0
    SGD  = 1


def get_optimizer(index, model, lr):
    if   index == Optimizer.ADAM:   return optim.Adam(model.parameters(), lr)
    elif index == Optimizer.SGD:    return optim.SGD(model.parameters(), lr)
