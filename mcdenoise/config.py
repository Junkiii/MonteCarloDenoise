# Hendrik Junkawitsch; Saarland University

# Configuration module
# Every crucial hyperparameter of a training run is set here

from loss import *
from scheduler import *
from models.modelchooser import *
from optimizer import *
from dataset import Aux

class Config():
    def __init__(self):
        # HYPERPARAMETER SETTING:
        self.num_epochs     = 1270
        self.batchsize      = 32
        self.lr             = 1e-4
        self.in_channels    = Aux.NDSN
        self.model_id       = Model.DAE_SKIP
        self.model          = get_model(self.model_id, self.in_channels)
        self.optimizer_id   = Optimizer.ADAM
        self.optimizer      = get_optimizer(self.optimizer_id, self.model, self.lr)
        self.loss_id        = Loss.MIX
        self.loss           = getLossById(self.loss_id)
        self.scheduler_id   = Schedule.CONST
        self.scheduler      = getSchedulerById(self.scheduler_id, self.optimizer, self.num_epochs)

        print()
        print("Configuration")
        print(">>> num_epochs ="     , self.num_epochs)
        print(">>> batch_size ="     , self.batchsize)
        print(">>> lr ="             , self.lr)
        print(">>> in_channels ="    , self.in_channels)
        print(">>> model_id ="       , self.model_id)
        print(">>> loss_id ="        , self.loss_id)
        print(">>> scheduler_id ="   , self.scheduler_id)
        print()

