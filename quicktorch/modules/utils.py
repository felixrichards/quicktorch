import torch.nn as nn


class RemovePadding(nn.Module):
    """Removes padding from tensor
    """
    def __init__(self, p):
        super().__init__()
        self.p = p // 2

    def forward(self, x):
        if self.p > 0:
            x = x[..., self.p:-self.p, self.p:-self.p]
        return x
