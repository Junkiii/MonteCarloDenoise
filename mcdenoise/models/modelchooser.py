# Hendrik Junkawitsch; Saarland University

# Model chooser module for the configuration module

from models.dae import DAE
from models.dae_skip import DAE_skip
from models.oidn import OIDN
from models.oidn_ae import OIDN_AE
from models.oidn_large_kernel import OIDN_lk
from models.oidn_small_cap import OIDN_SC
from models.oidn_small_cap2 import OIDN_SC2
from enum import IntEnum

class Model(IntEnum):
    DAE         = 1
    DAE_SKIP    = 2
    OIDN        = 3
    OIDN_AE     = 4
    OIDN_LK     = 5
    OIDN_SC     = 6
    OIDN_SC2    = 7


def get_model(model, in_channels):
    if   model == Model.DAE:        return DAE(in_channels=in_channels)
    elif model == Model.DAE_SKIP:   return DAE_skip(in_channels=in_channels)
    elif model == Model.OIDN:       return OIDN(in_channels=in_channels)
    elif model == Model.OIDN_AE:    return OIDN_AE(in_channels=in_channels)
    elif model == Model.OIDN_LK:    return OIDN_lk(in_channels=in_channels)
    elif model == Model.OIDN_SC:    return OIDN_SC(in_channels=in_channels)
    elif model == Model.OIDN_SC2:   return OIDN_SC2(in_channels=in_channels)
