import torch
import torch.nn as nn
from torchvision.models import resnext50_32x4d, resnet50


class ResNet50Features(nn.Module):
    def __init__(self, n_channels=4):
        super().__init__()
        self.n_channels = n_channels
        net = self.resnet50()
        self.layer0 = nn.Sequential(*net[:5])
        self.layer1 = net[5]
        self.layer2 = net[6]
        self.layer3 = net[7]

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        return layer3, layer2, layer1

    def resnet50(self):
        model = resnet50(pretrained=True)

        conv1 = model.conv1
        first_layer = nn.Conv2d(self.n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            first_layer.weight[:, :3] = conv1.weight
        model.conv1 = first_layer
        return list(model.children())[:8]

    def resnext50(self):
        model = resnext50_32x4d(pretrained=True)

        conv1 = model.conv1
        first_layer = nn.Conv2d(self.n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            first_layer.weight[:, :3] = conv1.weight
        model.conv1 = first_layer
        return list(model.children())[:8]


class StandardFeatures(nn.Module):
    def __init__(self, n_channels=4, base_channels=64):
        super().__init__()
        self.n_channels = n_channels
        self.layer0 = standard_block(n_channels, base_channels * 1)
        self.layer1 = standard_block(base_channels * 1, base_channels * 2)
        self.layer2 = standard_block(base_channels * 2, base_channels * 3)
        self.layer3 = standard_block(base_channels * 3, base_channels * 4)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


def standard_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )
