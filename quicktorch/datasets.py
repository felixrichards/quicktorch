from torchvision import transforms
from torch.utils.data.dataset import Dataset
import pandas as pd
import os
from skimage import io
import PIL.Image as Image
import glob
import numpy as np
from .customtransforms import MakeCategorical
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
