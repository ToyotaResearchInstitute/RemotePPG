import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
tr = torch
import torch.cuda as cutorch


class CalculateNormPSD(nn.Module):
    def __init__(self, Fs, high_pass, low_pass):
        super().__init__()
        self.Fs = Fs
        self.high_pass = high_pass
        self.low_pass = low_pass

    def forward(self, x, zero_pad=0):
        x = x - torch.mean(x)
        if zero_pad > 0:
            L = x.shape[-1]
            x = F.pad(x, (int(zero_pad/2*L), int(zero_pad/2*L)), 'constant', 0)

        # Get PSD
        x = torch.view_as_real(torch.fft.rfft(x, dim=-1, norm='forward'))
        x = tr.add(x[:, :, 0] ** 2, x[:, :, 1] ** 2)

        # Filter PSD for relevant parts
        Fn = self.Fs / 2
        freqs = torch.linspace(0, Fn, x.shape[1])
        use_freqs = torch.logical_and(freqs >= self.high_pass / 60, freqs <= self.low_pass / 60)
        x = x[:,use_freqs]

        # Normalize PSD
        x = x / torch.sum(x, dim=-1, keepdim=True)
        return x


class CalculateMultiView(nn.Module):
    def __init__(self, sub_length, num_views):
        super().__init__()
        self.num_views = num_views
        self.sub_length = sub_length
    def forward(self, input, zero_pad=0):
        # Pad input to be at least sub_length long
        if input.shape[-1] < self.sub_length:
            input = F.pad(input, (0, self.sub_length - input.shape[-1]))
        # Stack all random views
        views = []
        for i in range(self.num_views):
            # Random subset
            offset = torch.randint(0, input.shape[-1] - self.sub_length + 1, (1,), device=input.device)
            x = input[..., offset:offset + self.sub_length]
            views.append(x)
        return views


def torch_random_uniform(low, high):
    return ((high - low) * torch.rand(1)) + low


def torch_create_sine(freq, Fs, D, device):
    times = torch.arange(D, device=device) / Fs
    values = torch.sin(times * freq * 2 * np.pi)
    return values


def set_random_seed(seed):
    print(f'Setting random seed to {seed}')
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
