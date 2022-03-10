import pytest

import torch
import torch.nn as nn
import quicktorch.modules.attention.models as am
import quicktorch.modules.attention.attention as aa
from time import time, sleep

grid_params = [
        (1, 2, True, 1024),
        (2, 2, True, 1024),
        (3, 2, True, 1024),
        (4, 2, True, 1024),
        (1, 3, True, 1024),
        (2, 3, True, 1024),
        (3, 3, True, 1024),
        (4, 3, True, 1024),
        (1, 2, False, 1024),
        (2, 2, False, 1024),
        (3, 2, False, 1024),
        (4, 2, False, 1024),
        # pytest.param("6*9", 42, marks=pytest.mark.xfail)
    ]


@pytest.mark.parametrize("d, f, g, w", grid_params,)
def test_grid(d, f, g, w):
    # x = torch.ones(3, 5, w * f ** d * 2 ** d, w * f ** d * 2 ** d)
    x = torch.ones(3, 1, w, w)
    dis = aa.Assemble(d, f)
    redis = aa.Assemble(-d, f)

    xdis = dis(x)
    expected_shape = (3 * (f ** 2) ** d, 1, w // f ** d, w // f ** d)
    assert xdis.shape == expected_shape, f"disassemble broke {xdis.shape=} != {expected_shape=}"

    xredis = redis(xdis)
    expected_shape = x.shape
    assert xredis.shape == x.shape, f"reassemble broke {xredis.shape=} != {expected_shape=}"


@pytest.mark.parametrize("d, f, g, w", grid_params,)
def test_gridded_attention(d, f, g, w):
    scales = list(range(d))
    bc = 16
    x = [torch.ones(1, bc, w // 2 ** (s + 3), w // 2 ** (s + 3)).cuda() for s in scales]
    fused = torch.ones(1, bc, w // 2 ** 3,  w // 2 ** 3).cuda()

    if not g:
        scales = [0] * len(scales)
    sem_mod1 = aa.SemanticModule(bc * 2)
    sem_mod2 = aa.SemanticModule(bc * 2)
    attention_heads = nn.ModuleList([
        aa.StandardAttention(
            bc,
            disassembles=sc,
            scale_factor=f,
            semantic_module1=sem_mod1,
            semantic_module2=sem_mod2,
        )
        for sc in scales[::-1]
    ]).cuda()

    start = time()
    outs = []
    for att_head, feature in zip(attention_heads, x):
        outs.append(att_head(feature, fused))
    print(f'{(d, f, g)=} took {time()-start:.3f} seconds')
    sleep(.5)
    outs = None
    torch.cuda.empty_cache()
    sleep(.5)
