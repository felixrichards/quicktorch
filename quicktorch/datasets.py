import glob
import os
import shutil

from typing import Tuple

import torch
import numpy as np
import PIL.Image as Image
import scipy.io

from torchvision import transforms
from torch.utils.data.dataset import Dataset

from .customtransforms import MakeCategorical
from .utils import download

"""This module provides wrappers for loading custom datasets.
"""


class MNISTRot(Dataset):
    """Loads MNISTRot dataset from file.

    Will download and extract if it does not exist.

    After extracting the dataset is contained in an 'amat' file, in which
    data is stored as text where each line contains 28x28+1 floats. The first
    28x28 are the image, the last is the label.

    Args:
        dir (str, optional): Directory to load data from. Will download data into
            directory if it does not exist.
        test (bool, optional): Whether to load the testing dataset. Defaults to False.
        transforms (list, optional): Transforms to apply to images.
        indices (arraylike, optional): Indices of samples to form dataset from.
        onehot: (bool, optional): format to load targets in.
    """

    url = ('http://www.iro.umontreal.ca/'
           '~lisa/icml2007data/mnist_rotation_new.zip')
    dlname = 'mnist_rotation_new.zip'
    raw_train_name = 'mnist_all_rotation_normalized_float_train_valid.amat'
    raw_test_name = 'mnist_all_rotation_normalized_float_test.amat'
    train_file = 'train.pt'
    test_file = 'test.pt'

    def __init__(self, dir='../data/mnistrot', test=False, indices=None,
                 transform=None, onehot=True):
        self.dir = dir
        self.transform = transform
        self.test = test
        self.num_classes = 10
        if onehot:
            self.categorical = MakeCategorical(self.num_classes)
        else:
            self.categorical = None

        if not os.path.isdir(dir):
            os.mkdir(dir)
        if not os.path.isdir(os.path.join(dir, 'raw')):
            os.mkdir(os.path.join(dir, 'raw'))
        if not os.path.exists(os.path.join(dir, 'raw', self.dlname)):
            print('MNISTrot raw data not found. Attempting to download.')
            self.download()

        if self.test:
            data_file = self.test_file
        else:
            data_file = self.train_file

        if not os.path.isdir(os.path.join(dir, 'processed')):
            os.mkdir(os.path.join(dir, 'processed'))
        if not os.path.exists(os.path.join(dir, 'processed', data_file)):
            print('MNISTrot processed data not found. Attempting to create.')
            self.process()

        self.data, self.targets = torch.load(os.path.join(dir, 'processed', data_file))
        if indices is not None:
            self.data, self.targets = self.data[indices], self.targets[indices]
        self.targets = self.targets.to(torch.long)

    def __getitem__(self, i):
        img, target = self.data[i], self.targets[i]
        if self.categorical is not None:
            target = self.categorical(target)

        if self.transform is not None:
            img = Image.fromarray(img.numpy(), mode='L')
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    def download(self):
        print('Downloading')
        download(self.url, os.path.join(self.dir, 'raw'), extract=True)
        print('Done')

    def process(self):
        print('Processing')

        raw_names = (self.raw_test_name, self.raw_train_name)
        data_files = (self.test_file, self.train_file)

        for raw_name, data_file in zip(raw_names, data_files):
            data = np.loadtxt(os.path.join(self.dir, 'raw', raw_name))
            targets = data[:, -1]
            targets = targets
            imgs = np.delete(data, 28 * 28, 1)
            imgs = np.reshape(imgs, (imgs.shape[0], 28, 28))
            imgs = np.transpose(imgs, (0, 2 , 1))
            imgs = (255 * imgs).astype('uint8')

            targets = torch.tensor(targets)
            imgs = torch.tensor(imgs)

            with open(os.path.join(self.dir, 'processed', data_file), 'wb') as f:
                torch.save((imgs, targets), f)

        print('Done')


class BSD500(Dataset):
    """Loads BSD500 dataset from file.

    Will download and extract if it does not exist.
    Data is stored in .mat format.

    Args:
        dir (str, optional): Directory to load data from. Will download data into
            directory if it does not exist.
        test (bool, optional): Whether to load the testing dataset. Defaults to False.
        transforms (list, optional): Albumentation transforms to apply to images.
        indices (arraylike, optional): Indices of samples to form dataset from.
    """

    url = ('http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz')
    dlname = 'BSR_bsds500.tgz'
    data_path = "BSR/BSDS500/data"

    def __init__(self, dir='../data/bsd500', test=False, indices=None,
                 transform=None, landscape=False, padding=0):
        self.dir = dir
        self.transform = transform
        self.test = test
        self.landscape = landscape

        phase = 'test' if test else 'train'

        if not os.path.isdir(os.path.join(dir, 'processed')):
            print('BSD500 processed data not found. Attempting to create.')
            if not os.path.exists(os.path.join(dir, 'raw', self.dlname)):
                print('BSD500 raw data not found. Attempting to download.')
                if not os.path.isdir(os.path.join(dir, 'raw')):
                    os.makedirs(os.path.join(dir, 'raw'))
                self.download()
            self.process()

        self.img_paths = [
            img for img in glob.glob(os.path.join(self.dir, 'processed', phase, 'images', '*.jpg'))
        ]
        self.mask_paths = [
            img for img in glob.glob(os.path.join(self.dir, 'processed', phase, 'labels', '*.png'))
        ]
        if indices is not None:
            self.img_paths = [self.img_paths[i] for i in indices]
            self.mask_paths = [self.mask_paths[i] for i in indices]

        self.padding = padding

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = np.array(Image.open(self.img_paths[i]))
        label = np.array(Image.open(self.mask_paths[i]))

        if self.landscape:
            if img.shape[0] > img.shape[1]:
                img, label = np.rot90(img, -1), np.rot90(label, -1)

        if self.transform is not None:
            t = self.transform(image=img, mask=label)
            img = t['image']
            label = t['mask']
        label = np.expand_dims(label, axis=2)

        # This forces batches to have different irregular dimenions
        # TODO figure out collate function that can handle this
        # if rot: # Unrotate portrait images
        #     img, label = np.rot90(img, -1), np.rot90(label, -1)

        img = transforms.ToTensor()(img.copy())
        label = transforms.ToTensor()(label.copy())

        # albumentations workaround
        if self.padding > 0:
            label = remove_padding(label, self.padding)

        return img, label

    def __len__(self):
        return len(self.img_paths)

    def download(self):
        print('Downloading')
        download(self.url, os.path.join(self.dir, 'raw'), extract=True)
        print('Done')

    def process(self):
        print('Processing')
        os.makedirs(os.path.join(self.dir, 'processed', 'train', 'images'))
        os.makedirs(os.path.join(self.dir, 'processed', 'train', 'labels'))
        os.makedirs(os.path.join(self.dir, 'processed', 'test', 'images'))
        os.makedirs(os.path.join(self.dir, 'processed', 'test', 'labels'))

        from_tos = (
            ('train', 'train'),
            ('val', 'train'),
            ('test', 'test')
        )

        for from_dir, to_dir in from_tos:
            self._move_images(os.path.join(self.data_path, "images", from_dir), to_dir)
            self._convert_mat_to_png(os.path.join(self.data_path, "groundTruth", from_dir), to_dir)
        print('Done')

    def _move_images(self, from_dir, to_dir):
        for f in glob.glob(os.path.join(self.dir, 'raw', from_dir, "*.jpg")):
            shutil.copy(f, os.path.join(self.dir, 'processed', to_dir, 'images'))

    def _convert_mat_to_png(self, from_dir, to_dir, aggregation='weighted'):
        for f in glob.glob(os.path.join(self.dir, 'raw', from_dir, "*.mat")):
            name = os.path.split(f)[-1]
            name = name[:len(name)-4] + '.png'
            gts = scipy.io.loadmat(f)
            gts = gts['groundTruth']
            gts = np.array([gts[0, i][0, 0][1] for i in range(gts.shape[1])])
            if aggregation == 'weighted':
                gts = gts.mean(axis=0)
            gts = (gts * 255).astype('uint8')
            Image.fromarray(gts).save(os.path.join(self.dir, 'processed', to_dir, 'labels', name))


class EMDataset(Dataset):
    """Loads ISBI EM dataset from file.

    Args:
        img_dir (str): Path to dataset directory.
        transform (Trasform, optional): Albumentation transform(s) to apply
            to images/masks.
        aug_mult (int, optional): Factor to increase dataset by with
            augmentations.
        indices (arraylike, optional): Indices of data samples to form dataset
            from.
    """
    def __init__(self, img_dir,
                 transform=None, aug_mult=4, indices=None):
        self.em_paths = [
            img for img in glob.glob(os.path.join(img_dir, 'volume/*.png'))
        ]
        self.mask_paths = [
            img for img in glob.glob(os.path.join(img_dir, 'labels/*.png'))
        ]

        self.test = False
        if len(self.mask_paths) == 0 and len(self.em_paths) > 0:
            self.test = True

        self.transform = transform
        self.aug_mult = aug_mult
        if indices is not None:
            self.em_paths = [self.em_paths[i] for i in indices]
            if not self.test:
                self.mask_paths = [self.mask_paths[i] for i in indices]

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        i = i // self.aug_mult
        em = np.array(Image.open(self.em_paths[i]))
        if self.test:
            mask = np.zeros_like(em)
        else:
            mask = np.array(Image.open(self.mask_paths[i]))

        if self.transform is not None:
            t = self.transform(image=em, mask=mask)
            em = t['image']
            mask = t['mask']
        em = np.expand_dims(em, axis=2)
        mask = np.expand_dims(mask, axis=2)
        return (
            transforms.ToTensor()(em),
            transforms.ToTensor()(mask)
        )

    def __len__(self):
        return len(self.em_paths) * self.aug_mult


class SwimsegDataset(Dataset):
    """Loads Swimseg cloud segmentation dataset from file.

    Args:
        img_dir (str): Path to dataset directory.
        fold (str): What dataset phase to load. Choices are ('train', 'val', 'test').
        transform (Trasform, optional): Albumentation transform(s) to apply
            to images/masks.
        aug_mult (int, optional): Factor to increase dataset by with
            augmentations.
        padding (int, optional): 
        preload (bool, optional): 
    """
    means = (0.4808, 0.5728, 0.6849)
    stds = (0.2310, 0.1911, 0.1612)

    def __init__(self, img_dir, fold, transform=None, aug_mult=1,
                 padding=0, preload=True):
        super().__init__()
        self.img_paths = [
            img for img in glob.glob(os.path.join(img_dir, fold, '*.png'))
        ]
        self.mask_paths = [
            img for img in glob.glob(os.path.join(img_dir, fold + '_labels', '*.png'))
        ]

        self.transform = transform
        self.aug_mult = aug_mult
        self.norm_transform = transforms.Normalize(self.means, self.stds)
        self.padding = padding
        self.preload = preload
        if preload:
            self.preload_images()

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        i = i // self.aug_mult
        if self.preload:
            img = self.images[i]
            mask = self.masks[i]
        else:
            img = np.array(Image.open(self.img_paths[i]))
            mask = np.array(Image.open(self.mask_paths[i]).convert('L'))

        if self.transform is not None:
            t = self.transform(image=img, mask=mask)
            img = t['image']
            mask = t['mask']
        img = transforms.ToTensor()(img)
        mask = transforms.ToTensor()(mask)

        # albumentations workaround
        if self.padding > 0:
            mask = remove_padding(mask, self.padding)

        return (
            self.norm_transform(img),
            mask
        )

    def preload_images(self) -> None:
        self.images = np.stack([Image.open(path) for path in self.img_paths])
        self.masks = np.stack([Image.open(path).convert('L') for path in self.mask_paths])

    def __len__(self) -> int:
        return len(self.img_paths) * self.aug_mult


def remove_padding(t: torch.Tensor, p: int) -> torch.Tensor:
    return t[..., p//2:-p//2, p//2:-p//2]
