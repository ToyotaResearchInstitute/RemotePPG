import torch
import torch.nn as nn
tr = torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch.fft


class NegPearsonsCorrLoss(nn.Module):
    def __init__(self):
        super(NegPearsonsCorrLoss, self).__init__()

    def forward(self, x, y):
        vx = x - torch.mean(x, dim=-1, keepdim=True)
        vy = y - torch.mean(y, dim=-1, keepdim=True)
        vx2 = torch.sum(vx ** 2, dim=-1)
        vy2 = torch.sum(vy ** 2, dim=-1)
        corr = torch.sum(vx * vy, dim=-1) / (torch.sqrt(vx2) * torch.sqrt(vy2))
        return -corr
