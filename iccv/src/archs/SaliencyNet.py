import math

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetZoom(nn.Module):

    def __init__(self, block, layers, depth, channels=3, num_classes=1000):
        self.inplanes = 64
        super().__init__()
        self.depth = depth
        self.channels = channels
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        if self.channels < 3:
            self.conv1_act = nn.Conv2d(self.channels, 64, kernel_size=7, stride=2, padding=3,
                                       bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def copy_pretrained_conv1_weights(self):
        if self.channels == 1:
            weight = nn.Parameter(self.conv1.weight[:,1:2,:,:].clone())
            self.conv1_act.weight = weight
        elif self.channels == 2:
            weight = nn.Parameter(self.conv1.weight[:,0:2,:,:].clone())
            self.conv1_act.weight = weight
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.channels == 3:
            x = self.conv1(x)
        else:
            x = self.conv1_act(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if self.depth > 1:
            x = self.layer2(x)
        if self.depth > 2:
            x = self.layer3(x)
        if self.depth > 3:
            x = self.layer4(x)
        return x                                  


def saliency_network_resnet18(device, pretrained=True, depth=1, channels=3):
    """Constructs a partial ResNet-18 model, optionally pre-trained on ImageNet
    """
    model = ResNetZoom(BasicBlock, [2, 2, 2, 2], depth, channels)

    if pretrained:
        print('Loading pretrained saliency model')
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], \
                                                 model_dir="checkpoints/modelzoo/"),
                                                 strict=False)
    if channels < 3:
        model.copy_pretrained_conv1_weights()

    model = model.to(device)
    return model