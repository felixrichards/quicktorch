import pytest
import os

from quicktorch.datasets import MNISTRot, BSD500, EMDataset, SwimsegDataset


SWIMSEG_PATH = '../data/swimseg'


@pytest.mark.skipif(not os.path.exists('../data/mnistrot'), reason='Dataset not found')
def test_mnistrot_dataset():
    pass


@pytest.mark.skipif(not os.path.exists('../data/mnistrot'), reason='Dataset not found')
def test_bsd_dataset():
    pass


@pytest.mark.skipif(not os.path.exists('../data/mnistrot'), reason='Dataset not found')
def test_em_dataset():
    pass


@pytest.mark.skipif(not os.path.exists(SWIMSEG_PATH), reason='Dataset not found')
class TestSwimsegDataset():

    @pytest.mark.parametrize('fold', ('train', 'val', 'test'))
    def test_swimseg_compiles(self, fold):
        SwimsegDataset(os.path.join(SWIMSEG_PATH, 'swimseg'), fold, preload=False)

    @pytest.mark.parametrize('fold', ('train', 'val', 'test'))
    def test_swimseg_loads(self, fold):
        dataset = SwimsegDataset(os.path.join(SWIMSEG_PATH, 'swimseg'), fold, preload=False)
        assert len(dataset) > 0

        image, mask = dataset[0]
        assert image.shape == (3, 600, 600)
        assert mask.shape == (1, 600, 600)

    @pytest.mark.parametrize('fold', ('train', 'val', 'test'))
    def test_swimseg_padding(self, fold):
        dataset = SwimsegDataset(os.path.join(SWIMSEG_PATH, 'swimseg'), fold, preload=False, padding_to_remove=100)
        assert len(dataset) > 0
        assert dataset.padding == 100

        image, mask = dataset[0]
        assert image.shape == (3, 600, 600)
        assert mask.shape == (1, 500, 500)
