import torch
import torch.nn as nn
tr = torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch.fft

from src.losses.NegativeMaxCrossCov import NegativeMaxCrossCov


class NegativeMaxCrossCorr(nn.Module):
    def __init__(self, Fs, high_pass, low_pass):
        super(NegativeMaxCrossCorr, self).__init__()
        self.cross_cov = NegativeMaxCrossCov(Fs, high_pass, low_pass)

    def forward(self, preds, labels):
        denom = tr.std(preds, dim=-1) * tr.std(labels, dim=-1)
        cov = self.cross_cov(preds, labels)
        output = torch.zeros_like(cov)
        for ii in range(len(denom)):
            if denom[ii] > 0:
                output[ii] = cov[ii] / denom[ii]
        return output
