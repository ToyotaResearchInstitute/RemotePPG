import torch
import torch.nn as nn
tr = torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch.fft

from src.shared.torch_utils import CalculateNormPSD
from src.losses.NegativeMaxCrossCorr import NegativeMaxCrossCorr


class MultiViewTripletLoss(nn.Module):
    def __init__(self, Fs, D, high_pass, low_pass, mvtl_distance):
        super(MultiViewTripletLoss, self).__init__()
        self.norm_psd = None
        if 'PSD' in mvtl_distance:
            self.norm_psd = CalculateNormPSD(Fs, high_pass, low_pass)
        if 'MSE' in mvtl_distance:
            self.distance_func = nn.MSELoss(reduction = 'none')
        elif 'L1' in mvtl_distance:
            self.distance_func = nn.L1Loss(reduction = 'none')
        elif 'NegMCC' == mvtl_distance:
            self.distance_func = NegativeMaxCrossCorr(Fs, high_pass, low_pass)
        else:
            raise Exception(f"ERROR: Unknown distance metric {mvtl_distance}")

    def compare_view_lists(self, list_a, list_b):
        total_distance = 0.
        for i in range(len(list_a)):
            for j in range(len(list_b)):
                total_distance += self.distance_func(list_a[i], list_b[j])
        return total_distance

    def forward(self, branches):   
        # Calculate NormPSD for each branch, if needed
        num_temp_views = len(branches['anc'])
        if self.norm_psd is not None:
            for key in branches.keys():
                for temp_i in range(num_temp_views):
                    branches[key][temp_i] = self.norm_psd(branches[key][temp_i])

        # Tally the triplet loss
        pos_loss = self.compare_view_lists(branches['anc'], branches['pos'])
        neg_loss = self.compare_view_lists(branches['anc'], branches['neg'])
        return (pos_loss - neg_loss) / num_temp_views * num_temp_views
