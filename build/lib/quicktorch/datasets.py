import torch
import torchvision
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import os
from skimage import io
import glob
import numpy as np


# We assume labels are stored in a CSV file with the images in the same folder
class ClassificationDataset(Dataset):
    def __init__(self, csv_file, transform=transforms.ToTensor()):
        self.csv_file = csv_file
        self.image_dir = os.path.split(csv_file)[0]
        csv_data = pd.read_csv(csv_file)
        self.image_paths = [os.path.join(self.image_dir, img) for img in csv_data['imagename']]
        self.transform = transform
        
        if type(csv_data['label'][0]) is str:
            key_to_val = {lbl: idx for idx, lbl in enumerate(set(csv_data['label']))}
            self.labels = [key_to_val[lbl] for lbl in csv_data['label']]
        else:
            self.labels = csv_data['label']

        self.num_classes = len(set(self.labels))
    
    def __getitem__(self, i):
        image = io.imread(self.image_paths[i])
        image = self.transform(image)
        label = self.labels[i]
        return image, label
    
    def __len__(self):
        return len(self.image_paths)


class MaskDataset(Dataset):
    def __init__(self, image_dir, target_image_dir,
                 transform=transforms.ToTensor(), target_transform=transforms.ToTensor(), idx=slice(None)):
        img_list = np.array(glob.glob(os.path.join(image_dir, '*.png')))
        tar_img_list = np.array(glob.glob(os.path.join(target_image_dir, '*.png')))
        self.image_paths = sorted(img_list[idx])
        self.target_image_paths = sorted(tar_img_list[idx])
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