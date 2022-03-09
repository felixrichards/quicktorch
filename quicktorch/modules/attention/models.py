import sys

from collections import OrderedDict
from functools import reduce

import torch
import torch.nn.functional as F
import torch.nn as nn
import quicktorch.modules.attention.attention

from quicktorch.models import Model
from quicktorch.modules.utils import RemovePadding
from quicktorch.modules.attention.attention import (
    Reassemble,
    _DecoderBlock,
    SemanticModule,
    MultiConv,
    GuidedAttention
)
from quicktorch.modules.attention.utils import ResNet50Features, get_ms_backbone

import matplotlib.pyplot as plt


def get_attention_model(key):
    return getattr(sys.modules[__name__], f'Attention{key}')


def get_attention_head(key):
    return getattr(sys.modules[quicktorch.modules.attention.attention.__name__], f'{key}AttentionHead')


def get_attention_module(key):
    return getattr(sys.modules[quicktorch.modules.attention.attention.__name__], f'{key}Attention')


def create_attention_backbone(
    **kwargs,
):
    if type(kwargs['attention_head']) is str:
        kwargs['attention_head'] = get_attention_head(kwargs['attention_head'])
    kwargs['attention_mod'] = get_attention_module(kwargs['attention_mod'])
    return AttentionMS(**kwargs)


class AttModel(Model):
    def __init__(self, n_channels=1, base_channels=64, n_classes=1, pad_to_remove=64, scale=None, backbone='Standard',
                 attention_head='Dual', attention_mod='Guided', scales=3, scale_factor=2, ms_image=True, gridded=True,
                 **kwargs):
        super().__init__(**kwargs)

        if type(scales) is int:
            scales = list(range(scales))

        self.backbone_key = backbone
        self.features = get_ms_backbone(backbone)(
            n_channels=n_channels,
            base_channels=base_channels,
            ms_image=ms_image,
            scales=scales
        )

        self.attention_net = create_attention_backbone(
            in_channels=self.features.out_channels,
            base_channels=base_channels,
            scales=scales,
            scale_factor=scale_factor,
            attention_mod=attention_mod,
            attention_head=attention_head,
            gridded=gridded,
        )
        self.scales = scales
        self.preprocess = scale

        self.mask_generator = MSAttMaskGenerator(base_channels, n_classes, scales, pad_to_remove=pad_to_remove)

    def forward(self, images):
        if self.preprocess is not None:
            images = self.preprocess(images)

        # Create multiscale features
        x = self.features(images)

        # Generate rough segmentations with attention net
        segs, refined_segs, aux_outs = self.attention_net(x)

        # Upsize and refine
        segs, refined_segs = self.mask_generator(images, segs, refined_segs)

        if self.training:
            return (
                segs + refined_segs,
                aux_outs
            )
        else:
            return sum(refined_segs) / len(refined_segs)


class MSAttMaskGenerator(nn.Module):
    def __init__(self, base_channels, n_classes, scales=[0, 1, 2], pad_to_remove=64):
        super().__init__()
        self.predicts = nn.ModuleList([nn.Conv2d(base_channels, n_classes, kernel_size=1) for _ in range(len(scales))])
        self.refines = nn.ModuleList([nn.Conv2d(base_channels, n_classes, kernel_size=1) for _ in range(len(scales))])
        self.scales = scales
        self.strip = RemovePadding(pad_to_remove)

    def forward(self, images, segs, refined_segs):
        output_size = images.size()

        segs = [F.interpolate(seg, size=output_size[2:], mode='bilinear') for seg in segs]
        refined_segs = [F.interpolate(seg, size=output_size[2:], mode='bilinear') for seg in refined_segs]

        segs = [predict(seg) for predict, seg in zip(self.predicts, segs)]
        refined_segs = [refine(seg) for refine, seg in zip(self.refines, refined_segs)]

        segs = [self.strip(seg) for seg in segs]
        refined_segs = [self.strip(seg) for seg in refined_segs]

        return segs, refined_segs


class AttentionMS(Model):
    def __init__(self, in_channels=[256, 256, 256], base_channels=64,
                 scales=[0, 1, 2], scale_factor=2, attention_mod=GuidedAttention,
                 attention_head=None, rcnn=False, gridded=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.rcnn = rcnn
        self.scales = scales

        print(base_channels)
        self.standardises = nn.ModuleList([nn.Sequential(
            nn.Conv2d(inc, base_channels, kernel_size=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        ) for inc in in_channels])

        self.ups2 = nn.ModuleList([_DecoderBlock(base_channels * (max(scales) - sc + 1), base_channels * (max(scales) - sc + 1)) for sc in scales])
        self.ups1 = nn.ModuleList([_DecoderBlock(base_channels * (max(scales) - sc + 1), base_channels * (max(scales) - sc + 1)) for sc in scales])

        self.sem_mod1 = SemanticModule(base_channels * 2)
        self.sem_mod2 = SemanticModule(base_channels * 2)

        # Controls size of dividing grids in gridded attention
        disassembles = scales if gridded else [0] * len(scales)

        #  Stacked Attention: Tie SemanticModules across weights
        self.attention_heads = nn.ModuleList([
            attention_mod(
                base_channels,
                disassembles=d,
                scale_factor=scale_factor,
                semantic_module1=self.sem_mod1,
                semantic_module2=self.sem_mod2,
                attention_head=attention_head
            )
            for d in disassembles[::-1]
        ])

        self.fuse = MultiConv(len(scales) * base_channels, base_channels, False)

    def upscale(self, ups, segs):
        bc = segs[0].shape[1]  # original feature channels
        out = ups[-1](segs[-1])
        for i in self.scales[-2::-1]:
            out = ups[i](torch.cat((segs[i], out), dim=1))

        return [out[:, sc * bc: (sc + 1) * bc] for sc in self.scales]

    def forward(self, features):
        if type(features) is dict:
            features = list(features.values())
        # print(', '.join([f'{feature.size()=}' for feature in features]))

        features = [standardise(feature) for standardise, feature in zip(self.standardises, features)]
        # print(', '.join([f'{feature.size()=}' for feature in features]))

        # Align scales
        fused = self.fuse(torch.cat([
            features[0],
            *[
                F.interpolate(feature, features[0].size()[-2:], mode='bilinear') for feature in features[1:]
            ]
        ], 1))
        # print(f'{fused.size()=}')
        refined_segs, aux_outs = zip(*[
            att_head(feature, fused) for att_head, feature in zip(self.attention_heads, features)
        ])

        # print(', '.join([f'{feature.size()=}' for feature in features]))
        # print(', '.join([f'{refined_seg.size()=}' for refined_seg in refined_segs]))
        segs = self.upscale(self.ups1, features)
        refined_segs = self.upscale(self.ups2, refined_segs)
        # print(', '.join([f'{seg.size()=}' for seg in segs]))
        # print(', '.join([f'{refined_seg.size()=}' for refined_seg in refined_segs]))

        if self.rcnn:
            out = OrderedDict({
                s: torch.cat((seg, ref_seg), dim=1) for s, seg, ref_seg in zip(self.scales, segs, refined_segs)
            })
            out.update({'aux_outs': aux_outs})
            return out
        return segs, refined_segs, aux_outs


def plot_features(features):
    fig, ax = plt.subplots(len(features), 8, figsize=(10, 15))
    for axrow, d in zip(ax, features):
        for axi, down_och in zip(axrow, d):
            axi.imshow(down_och[0].detach().cpu().numpy())
