import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
import shutil
import numpy as np
import scipy.io as sio

from src.shared.torch_utils import torch_create_sine
import time

class MeanBaseline(nn.Module):
    """Residual Block."""
    def __init__(self, dataset, label_name='our'):
        super(MeanBaseline, self).__init__()

        self.Fs = dataset.options.target_freq
        means = []
        for sample in dataset:
            means.append(torch.mean(sample['targets'][label_name]))
        self.mean = torch.mean(torch.stack(means)) / 60

    def forward(self, x):
        sine_wave = torch_create_sine(self.mean, self.Fs, x.shape[2], x.device)
        # Make B x 1 x D x 1 x 1
        sine_wave = sine_wave.unsqueeze(0).repeat(x.shape[0], 1)
        sine_wave = sine_wave.unsqueeze(1).unsqueeze(3).unsqueeze(4)
        return sine_wave
