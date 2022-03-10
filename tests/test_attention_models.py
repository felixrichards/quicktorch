import torch
import quicktorch.modules.attention.models as am
from quicktorch.modules.attention.utils import get_ms_backbone


def test_attention_ms():
    x = [
        torch.rand(1, 256, 8 * 2 ** i, 8 * 2 ** i) for i in range(3, 0, -1)
    ]
    model = am.AttentionMS()
    out = model(x)
    assert len(out) == 3, "may be problem with attentionms"

    model = am.AttentionMS(ms_image=False)
    out = model(x)
    assert len(out) == 3, "attentionms ms_image=False broke"


def test_attention_nongridded():
    x = [
        torch.rand(1, 256, 8 * 2 ** i, 8 * 2 ** i) for i in range(3, 0, -1)
    ]
    model = am.AttentionMS(gridded=False)
    out = model(x)
    assert len(out) == 3, "standard non gridded broke"

    model = am.AttentionMS(ms_image=False, gridded=False)
    out = model(x)
    assert len(out) == 3, "ms_image=False non gridded broke"


def test_att_mask_generator():
    images = torch.rand(1, 1, 64, 64)
    x1 = [
        torch.rand(1, 64, 32, 32) for _ in range(3)
    ]
    x2 = [
        torch.rand(1, 64, 32, 32) for _ in range(3)
    ]
    model = am.MSAttMaskGenerator(64, 1)
    out = model(images, x1, x2)
    assert len(out) == 2 and out[0][0].shape[:-2] == images.shape[:-2], "AttMaskGenerator problem"


def test_backbones():
    images = torch.rand(1, 1, 256, 256)
    backbones = ['Standard', 'ResNet50']

    for b in backbones:
        backbone_cls = get_ms_backbone(b)
        for ms_image in [True, False]:
            model = backbone_cls(ms_image=ms_image, n_channels=1)
            out = model(images)
            assert len(out) == 3, f"Backbone f{b} len(out) is not 3"
            assert all([o.shape[1] == oc for o, oc in zip(out, model.out_channels)]), f"Backbone f{b} {[o.shape for o in out]} != {model.out_channels} with {ms_image=}"
