import torch
import torch.nn.functional as F
import torch.nn as nn


class _EncoderBlock(nn.Module):
    """
    Encoder block for Semantic Attention Module
    """
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    """
    Decoder Block for Semantic Attention Module
    """
    def __init__(self, in_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)


class SemanticModule(nn.Module):
    """
    Semantic attention module
    """
    def __init__(self, in_channels):
        super().__init__()
        self.chanel_in = in_channels

        self.enc1 = _EncoderBlock(in_channels, in_channels * 2)
        self.enc2 = _EncoderBlock(in_channels * 2, in_channels * 4)
        self.dec2 = _DecoderBlock(in_channels * 4, in_channels * 2)
        self.dec1 = _DecoderBlock(in_channels * 2, in_channels)

    def forward(self, x):

        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)

        dec2 = self.dec2(enc2)
        dec1 = self.dec1(F.upsample(dec2, enc1.size()[2:], mode='bilinear'))

        return enc2.view(-1), dec1


class PAM_Module(nn.Module):
    """ Position attention module"""
    def __init__(self, in_channels):
        super(PAM_Module, self).__init__()

        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.align = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Parameters:
        ----------
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.align(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_channels):
        super(CAM_Module, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.align = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Parameters:
        ----------
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)

        energy = torch.bmm(proj_query, proj_key)
        energy = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.align(energy)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class AttentionLayer(nn.Module):
    """
    Helper Function for PAM and CAM attention

    Parameters:
    ----------
    input:
        in_channels : input channels
        use_pam : Boolean value whether to use PAM_Module or CAM_Module
    output:
        returns the attention map
    """
    def __init__(self, in_channels, AttentionModule=PAM_Module):
        super().__init__()

        self.attn = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
            AttentionModule(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.PReLU()
        )

    def forward(self, x):
        return self.attn(x)


class MultiConv(nn.Module):
    """
    Helper function for Multiple Convolutions for refining.

    Parameters:
    ----------
    inputs:
        in_channels : input channels
        out_channels : output channels
        attn : Boolean value whether to use Softmax or PReLU
    outputs:
        returns the refined convolution tensor
    """
    def __init__(self, in_channels, out_channels, attn=True):
        super(MultiConv, self).__init__()

        self.fuse_attn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.Softmax2d() if attn else nn.PReLU()
        )

    def forward(self, x):
        return self.fuse_attn(x)


class Disassemble(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, xs = compress(x)
        x = disassemble(x)
        x = recover(x, xs)
        return x


class Reassemble(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, xs = compress(x)
        if xs is not None:
            xs = [xs[0] // 4, *xs[1:]]
        x = reassemble(x)
        x = recover(x, xs)
        return x


def disassemble(x):
    _, c, w, h = x.size()
    x = x.unfold(2, w // 2, w // 2)
    x = x.unfold(3, h // 2, h // 2)
    x = x.permute(0, 2, 3, 1, 4, 5)
    x = x.reshape(-1, c, w // 2, h // 2)
    return x


def reassemble(x):
    b, c, w, h = x.size()
    x = x.view(b // 4, 4, c, w, h)
    x = x.permute(0, 2, 3, 4, 1)
    x = x.reshape(b // 4, c * w * h, 4)
    x = F.fold(x, (w * 2, h * 2), (w, h), (1, 1), stride=(w, h))
    return x


def compress(x):
    xs = None
    if x.dim() == 5:
        xs = x.size()
        x = x.view(
            xs[0],
            xs[1] * xs[2],
            *xs[3:]
        )

    return x, xs


def recover(x, xs):
    if xs is not None:
        x = x.view(
            -1,
            *xs[1:3],
            x.size(-2),
            x.size(-1)
        )

    return x
