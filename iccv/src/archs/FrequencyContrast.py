import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import FiveCrop

from src.model_initialization import init_model
from src.shared.torch_utils import CalculateMultiView


class FrequencyContrast(nn.Module):
    """ Frequency contrast wrapper around a backbone model e.g. PhysNet
    """
    def __init__(self, args, device, dataset):
        super().__init__()

        self.backbone = init_model(args.contrast_model, args, device, dataset)
        self.upsampler = nn.Upsample(size=(dataset.options.D,), mode='linear', align_corners=False)
        self.get_temp_views = CalculateMultiView(args.mvtl_window_size, args.mvtl_number_views)

    def forward(self, x_a):
        B = x_a.shape[0]
        D = x_a.shape[2]
        branches = {}

        # Resample input
        freq_factor = 1.25 + (torch.rand(1, device=x_a.device) / 4)
        target_size = int(D / freq_factor)
        resampler = nn.Upsample(size=(target_size, x_a.shape[3], x_a.shape[4]),
                                mode='trilinear',
                                align_corners=False)
        x_n = resampler(x_a)
        x_n = F.pad(x_n, (0, 0, 0, 0, 0, D - target_size))

        # Pass both samples through backbone
        y_a = self.backbone(x_a).squeeze(4).squeeze(3)
        y_n = self.backbone(x_n).squeeze(4).squeeze(3)

        # Remove padding from negative branch
        y_n = y_n[:,:,:target_size]

        # Resample negative PPG to create positive branch
        y_p = self.upsampler(y_n)

        # Save branches and backbone
        branches['anc'] = y_a.squeeze(1)
        branches['neg'] = y_n.squeeze(1)
        branches['pos'] = y_p.squeeze(1)

        # Create backbone output
        backbone_out = branches['anc']

        # Sample random views for each branch
        for key, branch in branches.items():
            branches[key] = self.get_temp_views(branch)
        
        return backbone_out, branches
