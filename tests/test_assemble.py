import pytest

import torch
import quicktorch.modules.attention.attention as am


@pytest.mark.parametrize(
    "d,f,w",
    [
        (1, 2, 256),
        (2, 2, 256),
        (3, 2, 256),
        (4, 2, 256),
        (1, 3, 243),
        (2, 3, 243),
        (3, 3, 243),
        (4, 3, 243),
        # pytest.param("6*9", 42, marks=pytest.mark.xfail)
    ],
)
def test_grid(d, f, w):
    x = torch.ones(3, 5, w, w)
    dis = am.Assemble(d, f)
    redis = am.Assemble(-d, f)

    xdis = dis(x)
    expected_shape = (3 * (f ** 2) ** d, 5, w // f ** d, w // f ** d)
    assert xdis.shape == expected_shape, f"disassemble broke {xdis.shape=} != {expected_shape=}"

    xredis = redis(xdis)
    expected_shape = x.shape
    assert xredis.shape == x.shape, f"reassemble broke {xredis.shape=} != {expected_shape=}"
