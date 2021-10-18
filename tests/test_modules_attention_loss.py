import pytest
import torch
from quicktorch.modules.attention.loss import DAFLossMCML


def test_daflossmcml():
    criterion = DAFLossMCML()
    target, output = torch.zeros(1, 3, 4, 4), torch.zeros(1, 3, 4, 4)
    # 7 pixels total positive
    target[0, 0, 1:3, 1:3] = 1  # 4
    target[0, 1, 0:2, 0:1] = 1  # 2
    target[0, 2, 0, 3] = 1      # 1

    # 2 correct
    output[0, 0, 1, 1] = 1
    output[0, 1, 2, 1] = 1

    loss = criterion(output, target)
    assert False