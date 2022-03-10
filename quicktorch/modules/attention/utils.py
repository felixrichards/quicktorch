import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnext50_32x4d, resnet50


def get_ms_backbone(key):
    return getattr(sys.modules[__name__], f'{key}Features')


class MSBackbone(nn.Module):
    def __init__(self, ms_image=True, scales=[0, 1, 2]):
        super().__init__()
        self.ms_image = ms_image
        self.scales = scales

    def forward(self, x):
        if self.ms_image:
            downs = self.create_ms_copies(x)
            downs = [self.generate_features(down)[-1] for down in downs]
        else:
            downs = self.generate_features(x)
            downs = self.downscale_features(downs)
        return downs

    def generate_features(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        return [layer1, layer2, layer3]

    def create_ms_copies(self, x):
        return [F.interpolate(
            x,
            scale_factor=1/2**s,
            recompute_scale_factor=True,
            mode='bilinear'
        ) if s != 0 else x for s in self.scales]

    def downscale_features(self, downs):
        return [F.interpolate(
            down,
            scale_factor=1/4,
            recompute_scale_factor=True,
            mode='bilinear'
        ) for down in downs]


class ResNet50Features(MSBackbone):
    def __init__(self, n_channels=4, ms_image=True, scales=[0, 1, 2], **kwargs):
        super().__init__(ms_image=ms_image, scales=scales)
        self.n_channels = n_channels
        net = self.resnet50()
        self.layer0 = nn.Sequential(*net[:5])
        self.layer1 = net[5]
        self.layer2 = net[6]
        self.layer3 = net[7]
        if self.ms_image:
            self.out_channels = [512 * 4] * len(self.scales)
        else:
            self.out_channels = [512 * 1, 512 * 2, 512 * 4]

    def resnet50(self):
        model = resnet50(pretrained=True)
        model = self.recorrect_in_channels(model)
        return list(model.children())[:8]

    def resnext50(self):
        model = resnext50_32x4d(pretrained=True)
        model = self.recorrect_in_channels(model)
        return list(model.children())[:8]

    def recorrect_in_channels(self, model):
        conv1 = model.conv1
        first_layer = nn.Conv2d(self.n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            first_layer.weight[:, :min(self.n_channels, 3)] = conv1.weight[:, :min(self.n_channels, 3)]
        model.conv1 = first_layer
        return model


class StandardFeatures(MSBackbone):
    def __init__(self, n_channels=4, base_channels=64, ms_image=True, scales=[0, 1, 2]):
        super().__init__(ms_image=ms_image, scales=scales)
        self.n_channels = n_channels
        self.layer0 = standard_block(n_channels, base_channels * 1)
        self.layer1 = standard_block(base_channels * 1, base_channels * 2)
        self.layer2 = standard_block(base_channels * 2, base_channels * 3)
        self.layer3 = standard_block(base_channels * 3, base_channels * 4, pool=False)
        if self.ms_image:
            self.out_channels = [base_channels * 4] * len(self.scales)
        else:
            self.out_channels = [base_channels * 2, base_channels * 3, base_channels * 4]


def standard_block(in_c, out_c, pool=True):
    block = [
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    ]
    if pool:
        block.append(nn.MaxPool2d(2))
    return nn.Sequential(*block)
