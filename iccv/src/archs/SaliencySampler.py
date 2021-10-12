import argparse
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.utils as vutils


def make_gaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which can be thought of as an effective radius.
    """
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)


class SaliencySampler(nn.Module):

    def __init__(self, task_network, saliency_network, depth, device, task_input_size, saliency_input_size):
        super().__init__()
        
        paper_res_ratio = np.array(saliency_input_size) / 128
        paper_joint_ratio = np.mean(paper_res_ratio)
        self.task_net = task_network
        self.grid_size = np.rint(paper_res_ratio * 32).astype(np.int)
        self.padding_size = int(np.rint(32 * paper_joint_ratio))
        self.fwhm = 13 * paper_joint_ratio
        self.global_size = self.grid_size + 2 * self.padding_size
        self.input_size = saliency_input_size
        self.input_size_net = task_input_size
        saliency_net_depths = {1: 64, 2: 128, 3: 256, 4: 512}
        self.conv_last = nn.Conv2d(saliency_net_depths[depth], 1, kernel_size=1, padding=0, stride=1).to(device)

        gaussian_weights = torch.FloatTensor(make_gaussian(2 * self.padding_size + 1, fwhm=self.fwhm)).to(device)

        # Spatial transformer localization-network
        self.saliency_net = saliency_network
        self.filter = nn.Conv2d(1, 1, kernel_size= (2 * self.padding_size + 1, 2 * self.padding_size + 1), bias=False).to(device)
        self.filter.weight[0].data[:,:,:] = gaussian_weights

        self.P_basis = torch.zeros(2, self.global_size[0], self.global_size[1], device=device)
        for k in range(2):
            for i in range(self.global_size[0]):
                for j in range(self.global_size[1]):
                    self.P_basis[k,i,j] = k * (i-self.padding_size) / (self.grid_size[0]-1.0) + \
                                          (1.0-k)*(j-self.padding_size)/(self.grid_size[1]-1.0)

    def create_grid(self, x):
        P = torch.nn.Parameter(torch.zeros(1, 2, self.global_size[0], self.global_size[1], device=x.device), requires_grad=False)
        P[0,:,:,:] = self.P_basis
        P = P.expand(x.size(0), 2, self.global_size[0], self.global_size[1])

        x_cat = torch.cat((x, x), 1)
        p_filter = self.filter(x)
        x_mul = torch.mul(P, x_cat).view(-1, 1, self.global_size[0], self.global_size[1])
        all_filter = self.filter(x_mul).view(-1, 2, self.grid_size[0], self.grid_size[1])

        x_filter = all_filter[:,0,:,:].contiguous().view(-1, 1, self.grid_size[0], self.grid_size[1])
        y_filter = all_filter[:,1,:,:].contiguous().view(-1, 1, self.grid_size[0], self.grid_size[1])
        x_filter = x_filter / p_filter
        y_filter = y_filter / p_filter
        xgrids = x_filter*2-1
        ygrids = y_filter*2-1
        xgrids = torch.clamp(xgrids, min=-1, max=1)
        ygrids = torch.clamp(ygrids, min=-1, max=1)
        xgrids = xgrids.view(-1, 1, self.grid_size[0], self.grid_size[1])
        ygrids = ygrids.view(-1, 1, self.grid_size[0], self.grid_size[1])

        grid = torch.cat((xgrids, ygrids), 1)
        grid = nn.Upsample(size=self.input_size_net, mode='bilinear', align_corners=True)(grid)
        grid = torch.transpose(grid,1,2)
        grid = torch.transpose(grid,2,3)

        return grid

    def forward(self, x, p):
        # switch from BCDHW to [B*D]CHW
        num_batches = x.shape[0]
        x = x.permute(0,2,1,3,4)
        x = x.reshape(-1, *(x.size()[2:]))
        x_low = nn.AdaptiveAvgPool2d(self.input_size)(x)        

        xs = self.saliency_net(x_low)
        xs = nn.ReLU()(xs)
        xs = self.conv_last(xs)
        xs = nn.Upsample(size=(self.grid_size[0], self.grid_size[1]), mode='bilinear', align_corners=True)(xs)
        xs = xs.view(-1, self.grid_size[0] * self.grid_size[1])
        xs = nn.Softmax(dim=1)(xs)
        xs = xs.view(-1, 1, self.grid_size[0], self.grid_size[1])
        xs_hm = nn.ReplicationPad2d(self.padding_size)(xs)

        grid = self.create_grid(xs_hm)

        x_sampled = F.grid_sample(x, grid, align_corners=True)

        if random.random() > p:
            s = random.randint(64, 224)
            x_sampled = nn.AdaptiveAvgPool2d((s, s))(x_sampled)
            x_sampled = nn.Upsample(size=self.input_size_net, mode='bilinear', align_corners=True)(x_sampled)

        # switch from [B*D]CHW to BCDHW
        x_sampled = x_sampled.view(num_batches, -1, *(x_sampled.size()[1:]))
        x_sampled = x_sampled.permute(0,2,1,3,4)
        xs = xs.view(num_batches, -1, *(xs.size()[1:]))

        x = self.task_net(x_sampled)
        
        return x, (x_sampled, xs)