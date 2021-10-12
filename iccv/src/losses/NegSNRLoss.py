import torch
import torch.nn as nn
tr = torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch.fft


class NegSNRLoss(nn.Module):
    def __init__(self, Fs, high_pass, low_pass):
        super(NegSNRLoss, self).__init__()
        self.Fs = Fs
        self.high_pass = high_pass
        self.low_pass = low_pass

    def forward(self, outputs: tr.Tensor, targets: tr.Tensor):
        device = outputs.device
        if not outputs.is_cuda:
            torch.backends.mkl.is_available()

        N = outputs.shape[-1]
        pulse_band = tr.tensor([self.high_pass/60., self.low_pass/60.], dtype=tr.float32).to(device)
        f = tr.linspace(0, self.Fs/2, int(N/2)+1, dtype=tr.float32).to(device)

        min_idx = tr.argmin(tr.abs(f - pulse_band[0]))
        max_idx = tr.argmin(tr.abs(f - pulse_band[1]))

        outputs = outputs.view(-1, N)
        if targets.shape[-1] != 1:
            targets = tr.mean(targets, -1, keepdim=True)
        targets = targets.view(-1, 1) / 60.

        X = torch.view_as_real(torch.fft.rfft(outputs, dim=-1, norm='forward'))
        P1 = tr.add(X[:, :, 0]**2, X[:, :, 1]**2)                                   # One sided Power spectral density

        # calculate indices corresponding to refs
        ref_idxs = []
        for ref in targets:
            ref_idxs.append(tr.argmin(tr.abs(f-ref)))

        # calc SNR for each batch
        snrs = tr.empty((len(ref_idxs),), dtype=tr.float32)
        freq_num_in_pulse_range = max_idx-min_idx
        for count, ref_idx in enumerate(ref_idxs):
            pulse_freq_amp = P1[count, ref_idx]
            other_avrg = (tr.sum(P1[count, min_idx:ref_idx-1]) + tr.sum(P1[count, ref_idx+2:max_idx]))/(freq_num_in_pulse_range-3)
            snrs[count] = 10*tr.log10(pulse_freq_amp/other_avrg)
        return -tr.mean(snrs)
