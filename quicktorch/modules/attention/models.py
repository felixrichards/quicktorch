import torch
import torch.nn.functional as F
import torch.nn as nn

from quicktorch.models import Model	
from igcn.seg.scale import Scale, ScaleParallel
from quicktorch.modules.attention.attention import (
    Disassemble,
    Reassemble,
    _DecoderBlock,
    SemanticModule,
    PAM_Module,
    CAM_Module,
    AttentionLayer,
    MultiConv
)


class DAFMSPlain(Model):
    def __init__(self, n_channels=1, base_channels=64, n_classes=1, pad_to_remove=64, scale=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.preprocess = scale

        self.disassemble = Disassemble()
        self.reassemble = Reassemble()
        self.p = pad_to_remove // 2

        self.down1 = nn.Sequential(
            nn.Conv2d(n_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels * 2, base_channels * 2 ** 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2 ** 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2 ** 2, base_channels * 2 ** 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2 ** 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels * 2 ** 2, base_channels * 2 ** 3, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2 ** 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2 ** 3, base_channels * 2 ** 3, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2 ** 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(base_channels * 2 ** 3, base_channels, kernel_size=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(base_channels * 2 ** 3, base_channels, kernel_size=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(base_channels * 2 ** 3, base_channels, kernel_size=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        self.up2 = _DecoderBlock(base_channels, base_channels)
        self.up1 = _DecoderBlock(base_channels, base_channels)

        self.conv8_2 = nn.Conv2d(base_channels, base_channels, kernel_size=1)
        self.conv8_3 = nn.Conv2d(base_channels, base_channels, kernel_size=1)
        self.conv8_4 = nn.Conv2d(base_channels, base_channels, kernel_size=1)
        self.conv8_12 = nn.Conv2d(base_channels, base_channels, kernel_size=1)
        self.conv8_13 = nn.Conv2d(base_channels, base_channels, kernel_size=1)
        self.conv8_14 = nn.Conv2d(base_channels, base_channels, kernel_size=1)

        self.semanticModule_1_1 = SemanticModule(base_channels * 2)

        self.conv_sem_1_2 = nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1)
        self.conv_sem_1_3 = nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1)
        self.conv_sem_1_4 = nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1)

        #Dual Attention mechanism
        self.pam_attention_1_1 = AttentionLayer(base_channels, PAM_Module)
        self.cam_attention_1_1 = AttentionLayer(base_channels, CAM_Module)
        self.pam_attention_1_2 = AttentionLayer(base_channels, PAM_Module)
        self.cam_attention_1_2 = AttentionLayer(base_channels, CAM_Module)
        self.pam_attention_1_3 = AttentionLayer(base_channels, PAM_Module)
        self.cam_attention_1_3 = AttentionLayer(base_channels, CAM_Module)

        self.semanticModule_2_1 = SemanticModule(base_channels * 2)

        self.conv_sem_2_2 = nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1)
        self.conv_sem_2_3 = nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1)
        self.conv_sem_2_4 = nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1)

        self.pam_attention_2_1 = AttentionLayer(base_channels, PAM_Module)
        self.cam_attention_2_1 = AttentionLayer(base_channels, CAM_Module)
        self.pam_attention_2_2 = AttentionLayer(base_channels, PAM_Module)
        self.cam_attention_2_2 = AttentionLayer(base_channels, CAM_Module)
        self.pam_attention_2_3 = AttentionLayer(base_channels, PAM_Module)
        self.cam_attention_2_3 = AttentionLayer(base_channels, CAM_Module)

        self.fuse1 = MultiConv(3 * base_channels, base_channels, False)

        self.predict3 = nn.Conv2d(base_channels, n_classes, kernel_size=1)
        self.predict2 = nn.Conv2d(base_channels, n_classes, kernel_size=1)
        self.predict1 = nn.Conv2d(base_channels, n_classes, kernel_size=1)

        self.predict3_2 = nn.Conv2d(base_channels, n_classes, kernel_size=1)
        self.predict2_2 = nn.Conv2d(base_channels, n_classes, kernel_size=1)
        self.predict1_2 = nn.Conv2d(base_channels, n_classes, kernel_size=1)

    def forward(self, x):
        if self.preprocess is not None:
            x = self.preprocess(x)
        output_size = x.size()
        # Create downscaled copies
        down3 = x
        down2 = F.interpolate(
            x,
            scale_factor=1/2
        )
        down1 = F.interpolate(
            x,
            scale_factor=1/4
        )

        # Generate features
        # print(f'{down1.size()=}, {down2.size()=}, {down3.size()=}')
        down1 = self.down1(down1)
        down2 = self.down1(down2)
        down3 = self.down1(down3)

        # down2 = self.align2(down2)
        # down3 = self.align3(down3)
        # print(f'{down1.size()=}, {down2.size()=}, {down3.size()=}')

        # down2 = self.reassemble(down2)
        # down3 = self.reassemble(self.reassemble(down3))
        # print(f'{down1.size()=}, {down2.size()=}, {down3.size()=}')

        down1 = self.conv1_1(down1)
        down2 = self.conv1_2(down2)
        down3 = self.conv1_3(down3)
        # print(f'{down1.size()=}, {down2.size()=}, {down3.size()=}')

        # Align scales
        fuse1 = self.fuse1(torch.cat((
            down3,
            F.interpolate(down2, down3.size()[-2:]),
            F.interpolate(down1, down3.size()[-2:])
        ), 1))
        # print(f'{fuse1.size()=}')

        # fuse1_3 = self.disassemble(self.disassemble(fuse1))
        # fuse1_2 = self.disassemble(F.interpolate(fuse1, size=down2.size()[-2:]))
        # fuse1_1 = F.interpolate(fuse1, size=down1.size()[-2:])
        # print(f'{fuse1_1.size()=}, {fuse1_2.size()=}, {fuse1_3.size()=}')

        fused1_3 = torch.cat((
            self.disassemble(self.disassemble(down3)),
            self.disassemble(self.disassemble(fuse1))
        ), dim=1)
        # print(f'First attention: {fused1_3.size()=}')
        semVector_1_2, semanticModule_1_2 = self.semanticModule_1_1(fused1_3)
        attention1_3 = self.conv8_2(
            (
                self.pam_attention_1_3(fused1_3) +
                self.cam_attention_1_3(fused1_3)
            ) *
            self.conv_sem_1_2(semanticModule_1_2)
        )
        # print(f'{attention1_3.size()=}')

        fused2_3 = torch.cat((
            self.disassemble(self.disassemble(down3)),
            attention1_3 * self.disassemble(self.disassemble(fuse1))
        ), dim=1)
        # print(f'First refine: {fused2_3.size()=}')
        semVector_2_2, semanticModule_2_2 = self.semanticModule_2_1(fused2_3)
        refine3 = self.conv8_12(
            (
                self.pam_attention_2_3(fused2_3) +
                self.cam_attention_2_3(fused2_3)
            ) *
            self.conv_sem_2_2(semanticModule_2_2)
        )
        # print(f'{refine3.size()=}')

        fused1_2 = torch.cat((
            self.disassemble(down2),
            self.disassemble(F.interpolate(fuse1, size=down2.size()[-2:]))
        ), dim=1)
        # print(f'Second attention: {fused1_2.size()=}')
        semVector_1_3, semanticModule_1_3 = self.semanticModule_1_1(fused1_2)
        attention1_2 = self.conv8_3(
            (
                self.pam_attention_1_2(fused1_2) +
                self.cam_attention_1_2(fused1_2)
            ) *
            self.conv_sem_1_3(semanticModule_1_3)
        )
        # print(f'{attention1_2.size()=}')

        fused2_2 = torch.cat((
            self.disassemble(down2),
            attention1_2 * self.disassemble(F.interpolate(fuse1, size=down2.size()[-2:]))
        ), dim=1)
        # print(f'Second refine: {fused2_2.size()=}')
        semVector_2_3, semanticModule_2_3 = self.semanticModule_2_1(fused2_2)
        refine2 = self.conv8_13(
            (
                self.pam_attention_2_2(fused2_2) +
                self.cam_attention_2_2(fused2_2)
            ) *
            self.conv_sem_2_3(semanticModule_2_3)
        )
        # print(f'{refine2.size()=}')

        fused1_1 = torch.cat((
            down1,
            F.interpolate(fuse1, size=down1.size()[-2:])
        ), dim=1)
        # print(f'Third attention: {fused1_1.size()=}')
        semVector_1_4, semanticModule_1_4 = self.semanticModule_1_1(fused1_1)
        attention1_1 = self.conv8_4(
            (
                self.pam_attention_1_1(fused1_1) +
                self.cam_attention_1_1(fused1_1)
            ) *
            self.conv_sem_1_4(semanticModule_1_4)
        )
        # print(f'{attention1_1.size()=}')

        fused2_1 = torch.cat((
            down1,
            attention1_1 * F.interpolate(fuse1, size=down1.size()[-2:])
        ), dim=1)
        # print(f'Third refine: {fused2_1.size()=}')
        semVector_2_4, semanticModule_2_4 = self.semanticModule_2_1(fused2_1)
        refine1 = self.conv8_14(
            (
                self.pam_attention_2_1(fused2_1) +
                self.cam_attention_2_1(fused2_1)
            ) *
            self.conv_sem_2_4(semanticModule_2_4)
        )
        # print(f'{refine1.size()=}')

        # print(f'{down1.size()=}, {down2.size()=}, {down3.size()=}')
        predict3 = self.up1(down3)
        predict2 = self.up1(down2)
        predict1 = self.up1(down1)

        if self.p > 0:
            predict3 = predict3[..., self.p//4:-self.p//4, self.p//4:-self.p//4]
            predict2 = predict2[..., self.p//4:-self.p//4, self.p//4:-self.p//4]
            predict1 = predict1[..., self.p//4:-self.p//4, self.p//4:-self.p//4]
        # print(f'{predict1.size()=}, {predict2.size()=}, {predict3.size()=}')

        predict3 = self.predict3(predict3)
        predict2 = self.predict2(predict2)
        predict1 = self.predict1(predict1)
        # print(f'{predict1.size()=}, {predict2.size()=}, {predict3.size()=}')
        predict3 = F.interpolate(predict3, size=output_size[2:], mode='bilinear', align_corners=True)
        predict2 = F.interpolate(predict2, size=output_size[2:], mode='bilinear', align_corners=True)
        predict1 = F.interpolate(predict1, size=output_size[2:], mode='bilinear', align_corners=True)

        if self.p > 0:
            predict3 = predict3[..., self.p//4:-self.p//4, self.p//4:-self.p//4]
            predict2 = predict2[..., self.p//4:-self.p//4, self.p//4:-self.p//4]
            predict1 = predict1[..., self.p//4:-self.p//4, self.p//4:-self.p//4]

        refine3 = self.reassemble(self.reassemble(refine3))
        refine2 = self.reassemble(refine2)
        # print(f'{refine1.size()=}, {refine2.size()=}, {refine3.size()=}')

        predict3_2 = self.up2(refine3)
        predict2_2 = self.up2(refine2)
        predict1_2 = self.up2(refine1)
        # print(f'{predict1_2.size()=}, {predict2_2.size()=}, {predict3_2.size()=}')

        predict3_2 = self.predict3_2(predict3_2)
        predict2_2 = self.predict2_2(predict2_2)
        predict1_2 = self.predict1_2(predict1_2)
        # print(f'{predict1_2.size()=}, {predict2_2.size()=}, {predict3_2.size()=}')

        predict3_2 = F.interpolate(predict3_2, size=output_size[2:], mode='bilinear', align_corners=True)
        predict2_2 = F.interpolate(predict2_2, size=output_size[2:], mode='bilinear', align_corners=True)
        predict1_2 = F.interpolate(predict1_2, size=output_size[2:], mode='bilinear', align_corners=True)

        if self.training:
            return (
                (
                    semVector_1_2,
                    semVector_1_3,
                    semVector_1_4,
                ), (
                    semVector_2_2,
                    semVector_2_3,
                    semVector_2_4,
                ), (
                   fused1_1,
                   fused1_2,
                   fused1_3,
                   fused2_1,
                   fused2_2,
                   fused2_3,
                ), (
                   semanticModule_1_4,
                   semanticModule_1_3,
                   semanticModule_1_2,
                   semanticModule_2_4,
                   semanticModule_2_3,
                   semanticModule_2_2,
                ), (
                   predict1,
                   predict2,
                   predict3,
                   predict1_2,
                   predict2_2,
                   predict3_2,
                )
            )
        else:
            return ((predict1_2 + predict2_2 + predict3_2) / 3)
