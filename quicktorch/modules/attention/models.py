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
    def __init__(self, n_channels=1, base_channels=64, n_classes=1, pad_to_remove=64, scale=None, backbone='MS',
                 attention_head='Dual', attention_mod='Guided', scales=3, ms_image=True,
                 **kwargs):
        super().__init__(**kwargs)

        if type(scales) is int:
            scales = list(range(scales))

        self.backbone_key = backbone
        self.features = get_ms_backbone(backbone)(n_channels, base_channels, ms_image=ms_image, scales=scales)

        self.attention_net = create_attention_backbone(
            in_channels=self.features.out_channels,
            base_channels=base_channels,
            scales=scales,
            attention_mod=attention_mod,
            attention_head=attention_head,
        )
        self.scales = scales
        self.preprocess = scale

        self.mask_generator = MSAttMaskGenerator(base_channels, n_classes, scales, pad_to_remove=64)

    def forward(self, x):
        if self.preprocess is not None:
            x = self.preprocess(x)

        # Create multiscale features
        x = self.features(x)

        # Generate rough segmentations with attention net
        segs, refined_segs, aux_outs = self.attention_net(x)

        # Upsize and refine
        segs, refined_segs = self.mask_generator(x, segs, refined_segs)

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
                 scales=[0, 1, 2], attention_mod=GuidedAttention,
                 attention_head=None, rcnn=False, ms_image=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.rcnn = rcnn
        self.scales = scales

        self.reassemble = Reassemble()

        print(base_channels)
        self.standardises = nn.ModuleList([nn.Sequential(
            nn.Conv2d(inc, base_channels, kernel_size=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        ) for inc in in_channels])

        self.ups2 = nn.ModuleList([_DecoderBlock(base_channels * (2 - sc + 1), base_channels * (2 - sc + 1)) for sc in scales])
        self.ups1 = nn.ModuleList([_DecoderBlock(base_channels * (2 - sc + 1), base_channels * (2 - sc + 1)) for sc in scales])

        self.sem_mod1 = SemanticModule(base_channels * 2)
        self.sem_mod2 = SemanticModule(base_channels * 2)

        #  Stacked Attention: Tie SemanticModules across weights
        self.attention_heads = nn.ModuleList([
            attention_mod(
                base_channels,
                disassembles=s,
                semantic_module1=self.sem_mod1,
                semantic_module2=self.sem_mod2,
                attention_head=attention_head
            )
            for s in scales[::-1]
        ])

        self.fuse = MultiConv(len(scales) * base_channels, base_channels, False)

    def upscale(self, ups, segs):
        bc = segs[0].shape[1]
        out = ups[2](segs[2])
        out = ups[1](torch.cat((segs[1], out), dim=1))
        out = ups[0](torch.cat((segs[0], out), dim=1))
        return [out[:, sc * bc: (sc + 1) * bc] for sc in self.scales]

    def forward(self, downs):
        # print(', '.join([f'{down.size()=}' for down in downs]))
        if type(downs) is dict:
            downs = list(downs.values())

        downs = [standardise(down) for standardise, down in zip(self.standardises, downs)]
        # print(', '.join([f'{down.size()=}' for down in downs]))

        # Align scales
        fused = self.fuse(torch.cat([
            downs[0],
            *[
                F.interpolate(down, downs[0].size()[-2:], mode='bilinear') for down in downs[1:]
            ]
        ], 1))
        # print(f'{fused.size()=}')
        refined_segs, aux_outs = zip(*[
            att_head(down, fused) for att_head, down in zip(self.attention_heads, downs)
        ])

        # print(', '.join([f'{down.size()=}' for down in downs]))
        # print(', '.join([f'{refined_seg.size()=}' for refined_seg in refined_segs]))
        # segs = [reduce(lambda r, f: f(r), self.ups1[:sc+1], down) for sc, down in zip(self.scales, downs)]
        # refined_segs = [reduce(lambda r, f: f(r), self.ups2[:sc+1], seg) for sc, seg in zip(self.scales, refined_segs)]
        segs = self.upscale(self.ups1, downs)
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


class AttentionResNet(Model):
    def __init__(self, n_channels=1, base_channels=64, scale=None, rcnn=False, ms_image=True, scales=[0, 1, 2],
                 **kwargs):
        super().__init__(**kwargs)
        self.preprocess = scale
        self.rcnn = rcnn
        self.scales = [0, 1, 2]

        self.reassemble = Reassemble()

        self.features = ResNet50Features(n_channels, ms_image=ms_image, scales=scales)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(512, base_channels, kernel_size=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(1024, base_channels, kernel_size=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(2048, base_channels, kernel_size=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        self.up2 = _DecoderBlock(base_channels, base_channels)
        self.up1 = _DecoderBlock(base_channels, base_channels)

        self.sem_mod1 = SemanticModule(base_channels * 2)
        self.sem_mod2 = SemanticModule(base_channels * 2)

        #  Stacked Attention: Tie SemanticModules across weights
        self.guided_attention1 = GuidedAttention(base_channels, 0, self.sem_mod1, self.sem_mod2)
        self.guided_attention2 = GuidedAttention(base_channels, 1, self.sem_mod1, self.sem_mod2)
        self.guided_attention3 = GuidedAttention(base_channels, 2, self.sem_mod1, self.sem_mod2)

        self.fuse = MultiConv(3 * base_channels, base_channels, False)

    def forward(self, x):
        if self.preprocess is not None:
            x = self.preprocess(x)

        # Generate features
        down1, down2, down3 = self.features(x)
        down1, down2, down3 = self.pool(down1), self.pool(down2), self.pool(down3)

        down1 = self.conv1_1(down1)
        down2 = self.conv1_2(down2)
        down3 = self.conv1_3(down3)
        # print(f'{down1.size()=}, {down2.size()=}, {down3.size()=}')

        # Align scales
        fused = self.fuse(torch.cat((
            down3,
            F.interpolate(down2, down3.size()[-2:], mode='bilinear'),
            F.interpolate(down1, down3.size()[-2:], mode='bilinear')
        ), 1))
        # print(f'{fused.size()=}')

        refine3, aux_outs3 = self.guided_attention3(down3, fused)
        refine2, aux_outs2 = self.guided_attention2(down2, fused)
        refine1, aux_outs1 = self.guided_attention1(down1, fused)

        # print(f'{down1.size()=}, {down2.size()=}, {down3.size()=}')
        predict3 = self.up1(down3)
        predict2 = self.up1(down2)
        predict1 = self.up1(down1)
        # print(f'{predict1.size()=}, {predict2.size()=}, {predict3.size()=}')

        refine3 = self.up2(refine3)
        refine2 = self.up2(refine2)
        refine1 = self.up2(refine1)
        # print(f'{refine1.size()=}, {refine2.size()=}, {refine3.size()=}')

        aux_outs = (aux_outs1, aux_outs2, aux_outs3)
        if self.rcnn:
            return OrderedDict({
                '0': torch.cat((predict1, refine1), dim=1),
                '1': torch.cat((predict2, refine2), dim=1),
                '2': torch.cat((predict3, refine3), dim=1),
                'aux_outs': aux_outs,
            })
        return (
            predict1,
            predict2,
            predict3,
        ), (
            refine1,
            refine2,
            refine3,
        ), aux_outs


def plot_features(features):
    fig, ax = plt.subplots(len(features), 8, figsize=(10, 15))
    for axrow, d in zip(ax, features):
        for axi, down_och in zip(axrow, d):
            axi.imshow(down_och[0].detach().cpu().numpy())
