import glob
import os
import shutil
import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import pandas as pd
from skimage import io
import PIL.Image as Image
import numpy as np
from .customtransforms import MakeCategorical
from .utils import download
import scipy.io
"""This module provides wrappers for loading custom datasets.
"""


class ClassificationDataset(Dataset):
    """Loads a classification dataset from file.

    Assumes labels are stored in a CSV file with the images in the same folder.
    It seems a little unintuitive and unnecessarily restrictive to support only
    passing a CSV filename for initialisation. Perhaps I will change this at
    some point.

    Args:
        csv_file (str, optional): Filename of csv file.
            Extension not necessary.
        transform (torchvision.transforms.Trasform, optional): Transform(s) to
        be applied to the data.
        **kwargs:
            weights_url (str, optional): A URL to download pre-trained weights.
            name (str, optional): See above. Defaults to None.
    """
    def __init__(self, csv_file, transform=transforms.ToTensor()):
        self.csv_file = csv_file
        self.image_dir = os.path.split(csv_file)[0]
        csv_data = pd.read_csv(csv_file)
        self.image_paths = [os.path.join(self.image_dir, img)
                            for img in csv_data['imagename']]

        if type(csv_data['label'][0]) is str:
            key_to_val = {lbl: idx
                          for idx, lbl in enumerate(set(csv_data['label']))}
            self.labels = [key_to_val[lbl] for lbl in csv_data['label']]
        else:
            self.labels = csv_data['label']

        self.num_classes = len(set(self.labels))
        self.transform = transform
        self.target_transform = MakeCategorical(n_classes=self.num_classes)

    def __getitem__(self, i):
        image = Image.open(self.image_paths[i])
        image = self.transform(image)
        label = self.target_transform(self.labels[i])
        return image, label

    def __len__(self):
        return len(self.image_paths)


class MaskDataset(Dataset):
    """Loads a mask dataset from file.

    Args:
        image_dir (str): Directory of training images.
        target_image_dir (str): Directory of image targets.
        transform (torchvision.transforms.Trasform, optional): Transform(s) to
            be applied to the training data.
            Defaults to transforms.ToTensor().
        target_transform (torchvision.transforms.Trasform, optional):
            Transform(s) to be applied to the target data.
            Defaults to transforms.ToTensor().
        idx (np.array, optional): Array of indices to select dataset.
            E.g. for folds.
        **kwargs:
            weights_url (str, optional): A URL to download pre-trained weights.
            name (str, optional): See above. Defaults to None.
    """
    def __init__(self, image_dir, target_image_dir,
                 transform=transforms.ToTensor(),
                 target_transform=transforms.ToTensor(), idx=slice(None)):
        imgs = np.array(glob.glob(os.path.join(image_dir, '*.png')))
        t_imgs = np.array(glob.glob(os.path.join(target_image_dir, '*.png')))
        self.image_paths = sorted(imgs[idx])
        self.target_image_paths = sorted(t_imgs[idx])
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        image = io.imread(self.image_paths[i])
        target = np.expand_dims(io.imread(self.target_image_paths[i]), axis=2)
        image = self.transform(image)
        target = self.target_transform(target)
        return image, target

    def __len__(self):
        return len(self.image_paths)


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
    """

    url = ('http://www.iro.umontreal.ca/'
           '~lisa/icml2007data/mnist_rotation_new.zip')
    dlname = 'mnist_rotation_new.zip'
    raw_train_name = 'mnist_all_rotation_normalized_float_train_valid.amat'
    raw_test_name = 'mnist_all_rotation_normalized_float_test.amat'
    train_file = 'train.pt'
    test_file = 'test.pt'

    def __init__(self, dir='../data/mnistrot', test=False, indices=None,
                 transform=None):
        self.dir = dir
        self.transform = transform
        self.test = test

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

    def __getitem__(self, i):
        img, target = self.data[i], MakeCategorical()(self.targets[i])

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
                 transform=None):
        self.dir = dir
        self.transform = transform
        self.test = test

        if not os.path.isdir(dir):
            os.mkdir(dir)
        if not os.path.isdir(os.path.join(dir, 'raw')):
            os.mkdir(os.path.join(dir, 'raw'))
        print(os.path.join(dir, 'raw', self.dlname))
        if not os.path.exists(os.path.join(dir, 'raw', self.dlname)):
            print('BSD500 raw data not found. Attempting to download.')
            self.download()

        phase = 'test' if test else 'train'

        if not os.path.isdir(os.path.join(dir, 'processed')):
            print('BSD500 processed data not found. Attempting to create.')
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

    def __getitem__(self, i):
        img = np.array(Image.open(self.img_paths[i]))
        label = np.array(Image.open(self.mask_paths[i]))

        rot = False # Rotate to normalise portrait to landscape
        if img.shape[0] > img.shape[1]:
            rot = True
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

        return (
            transforms.ToTensor()(img.copy()),
            transforms.ToTensor()(label.copy())
        )

    def __len__(self):
        return len(self.img_paths)

    def download(self):
        print('Downloading')
        download(self.url, os.path.join(self.dir, 'raw'), extract=True)
        print('Done')

    def process(self):
        print('Processing')
        os.mkdir(os.path.join(self.dir, 'processed'))
        os.mkdir(os.path.join(self.dir, 'processed', 'train'))
        os.mkdir(os.path.join(self.dir, 'processed', 'test'))
        os.mkdir(os.path.join(self.dir, 'processed', 'train', 'images'))
        os.mkdir(os.path.join(self.dir, 'processed', 'train', 'labels'))
        os.mkdir(os.path.join(self.dir, 'processed', 'test', 'images'))
        os.mkdir(os.path.join(self.dir, 'processed', 'test', 'labels'))

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
            gts = np.array([gts[0,i][0,0][1] for i in range(gts.shape[1])])
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

    def __getitem__(self, i):
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
