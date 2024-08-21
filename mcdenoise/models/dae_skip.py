# Hendrik Junkawitsch; Saarland University

import torch 
import torch.nn as nn
import torch.nn.functional as F

def Conv(i, o):
    return nn.Conv2d(in_channels=i, out_channels=o, kernel_size=3, padding=1)

class DAE_skip(nn.Module):
    def __init__(self, in_channels=9, out_channels=3):
        super(DAE_skip, self).__init__()

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2,2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.concat = lambda a,b: torch.cat((a,b),1)

        # fibonacci series channel numbers
        c1 = 21
        c2 = 34
        c3 = 55
        c4 = 89
        c5 = 144

        self.econv1 = Conv(in_channels, c1)
        self.econv2 = Conv(c1, c2)
        self.econv3 = Conv(c2, c3)
        self.econv4 = Conv(c3, c4)
        self.econv5 = Conv(c4, c5)

        self.dconv5 = Conv(c5, c4)
        self.dconv4 = Conv(c4*2, c3)
        self.dconv3 = Conv(c3*2, c2)
        self.dconv2 = Conv(c2*2, c1)
        self.dconv1 = Conv(c1*2, out_channels)

    def forward(self, x):
        x = self.relu(self.econv1(x))
        skip1 = x
        x = self.pool(x)

        x = self.relu(self.econv2(x))
        skip2 = x
        x = self.pool(x)

        x = self.relu(self.econv3(x))
        skip3 = x
        x = self.pool(x)

        x = self.relu(self.econv4(x))
        skip4 = x
        x = self.pool(x)

        x = self.relu(self.econv5(x))
        x = self.relu(self.dconv5(x))

        x = self.upsample(x)
        x = self.concat(x, skip4)
        x = self.relu(self.dconv4(x))

        x = self.upsample(x)
        x = self.concat(x, skip3)
        x = self.relu(self.dconv3(x))

        x = self.upsample(x)
        x = self.concat(x, skip2)
        x = self.relu(self.dconv2(x))

        x = self.upsample(x)
        x = self.concat(x, skip1)
        x = self.relu(self.dconv1(x))

        return x

