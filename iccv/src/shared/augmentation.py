import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision.transforms import RandomRotation, ToPILImage, ToTensor, ColorJitter, Resize
import torchvision.transforms.functional as TF
import torch.nn as nn
tr = torch

from src.shared.torch_utils import torch_random_uniform


class ImageAugmentation(nn.Module):
    def __init__(self, frame_height, frame_width, W=1.):
        super().__init__()
        self.W = W
        nW = 1. - W
        self.h_flip = torch_random_uniform(0., 1.) < (0.5 * W)

    def forward(self, img):
        if self.W == 0:
            return img

        img = ToPILImage()(img)
        if self.h_flip:
            img = TF.hflip(img)
        img = ToTensor()(img)
        return img
