import torch
import torchvision
from torchvision import transforms
from .customtransforms import ConvertType, MakeCategorical
from .datasets import MaskDataset, MNISTRot, BSD500
"""
This module contains functions that heavily abstract the
dataset loading process for some famous datasets.
"""


def cifar(alexnet=False, batch_size=4, hundred=False):
    """
    Loads the CIFAR10 dataset.

    This function will search for the dataset first in ./data.
    If the dataset is not found the function will attempt to download it.
    Args:
        alexnet (boolean, optional): If true the images will be resized to
            224x224 as in the AlexNet paper.
        batch_size (int, optional): Batch size for the DataLoader.
            Defaults to 4.

    Returns:
        torch.utils.data.DataLoader: Contains the training dataset.
        torch.utils.data.DataLoader: Contains the testing dataset.
        tuple: Class labels.
    """
    default_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    if alexnet:
        default_transforms.insert(0, transforms.Resize(size=(224, 224)))

    transform = transforms.Compose(default_transforms)
    target_transform = transforms.Compose([
        ConvertType(torch.float),
        MakeCategorical(10)
    ])

    if hundred:
        Dataset = torchvision.datasets.CIFAR100
    else:
        Dataset = torchvision.datasets.CIFAR10

    trainset = Dataset(root='./data', train=True,
                       download=True, transform=transform,
                       target_transform=target_transform)
    testset = Dataset(root='./data', train=False,
                      download=True, transform=transform,
                      target_transform=target_transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False)

    if not hundred:
        classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    else:
        classes = None

    return trainloader, testloader, classes


def mnist(dir='../data/mnist', batch_size=32, rotate=False, num_workers=0):
    """
    Loads the MNIST dataset.

    Args:
        batch_size (int, optional): Batch size for the DataLoader.
            Defaults to 32.
        rotate (boolean, optional): If true will apply random rotation
            to each sample.
        num_workers (int, optional): Number of workers given to the DataLoader.

    Returns:
        torch.utils.data.DataLoader: Contains the training dataset.
        torch.utils.data.DataLoader: Contains the testing dataset.
        tuple: Class labels.
    """
    transform = [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]
    if rotate:
        if torchvision.__version__[:3] == '0.5':
            transform.insert(0, transforms.RandomRotation(180, fill=(0,)))
        else:
            transform.insert(0, transforms.RandomRotation(180))

    transform = transforms.Compose(transform)
    target_transform = MakeCategorical()

    trainloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(dir, train=True, download=True,
                                   transform=transform,
                                   target_transform=target_transform),
        batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(dir, train=False, transform=transform,
                                   target_transform=target_transform),
        batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=num_workers)

    classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")

    return trainloader, testloader, classes

def mnistrot(batch_size=32, num_workers=0, transform=None, dir='../data/mnistrot/',
             test=False, split=None, rotate=False):
    """
    Loads the MNISTrot dataset.

    Args:
        batch_size (int, optional): Batch size for the DataLoader.
            Defaults to 32.
        num_workers (int, optional): Number of workers given for DataLoader.
        dir (str, optional): Directory to load data from. Will download
            data into directory if it does not exist.
        test (bool, optional): Whether to load the testing dataset.
            Defaults to False.
        transforms (list, optional): Transforms to apply to images.
        split (bool, optional): split indices used to generate
            training/validation samples.

    Returns:
        torch.utils.data.DataLoader: Contains the training dataset.
        torch.utils.data.DataLoader: Contains the testing dataset.
        tuple: Class labels.
    """
    classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
    transform = [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]

    if rotate:
        if torchvision.__version__[:3] == '0.5':
            transform.insert(0, transforms.RandomRotation(180, fill=(0,)))
        else:
            transform.insert(0, transforms.RandomRotation(180))

    transform = transforms.Compose(transform)
    if test:
        dataloader = torch.utils.data.DataLoader(
            MNISTRot(dir, test=True, transform=transform),
            batch_size=batch_size, shuffle=True,
            pin_memory=True, num_workers=num_workers
        )
        return dataloader, classes
    else:
        if split is None:
            split = [
                torch.arange(10000),
                torch.arange(10000, 12000)
            ]
        trainloader = torch.utils.data.DataLoader(
            MNISTRot(dir, test=False, indices=split[0], transform=transform),
            batch_size=batch_size, shuffle=True,
            pin_memory=True, num_workers=num_workers
        )
        testloader = torch.utils.data.DataLoader(
            MNISTRot(dir, test=False, indices=split[1], transform=transform),
            batch_size=batch_size, shuffle=True,
            pin_memory=True, num_workers=num_workers
        )
        return trainloader, testloader, classes


def bsd(batch_size=32, num_workers=0, transform=None, dir='../data/bsd500/',
             test=False, split=None):
    """
    Loads the BSD500 dataset.

    Args:
        batch_size (int, optional): Batch size for the DataLoader.
            Defaults to 32.
        num_workers (int, optional): Number of workers given for DataLoader.
        transform (list, optional): Transforms to apply to images.
        dir (str, optional): Directory to load data from. Will download
            data into directory if it does not exist.
        test (bool, optional): Whether to load the testing dataset.
            Defaults to False.
        split (bool, optional): split indices used to generate
            training/validation samples.

    Returns:
        torch.utils.data.DataLoader: Contains the training/testing dataset.
        torch.utils.data.DataLoader: Contains the validation dataset if test==false.
    """
    transform = transforms.Compose(transform)
    if test:
        dataloader = torch.utils.data.DataLoader(
            BSD500(dir, test=True, transform=transform),
            batch_size=batch_size, shuffle=True,
            pin_memory=True, num_workers=num_workers
        )
        return dataloader
    else:
        if split is None:
            split = [
                torch.arange(250),
                torch.arange(250, 300)
            ]
        trainloader = torch.utils.data.DataLoader(
            BSD500(dir, test=False, indices=split[0], transform=transform),
            batch_size=batch_size, shuffle=True,
            pin_memory=True, num_workers=num_workers
        )
        testloader = torch.utils.data.DataLoader(
            BSD500(dir, test=False, indices=split[1], transform=transform),
            batch_size=batch_size, shuffle=True,
            pin_memory=True, num_workers=num_workers
        )
        return trainloader, testloader


def clouds(image_dir='./clouds/swimseg/images/',
           target_image_dir='./clouds/swimseg/GTmaps/',
           train_idx=None, test_idx=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    cloud_train = MaskDataset(image_dir, target_image_dir, idx=train_idx,
                              transform=transform, target_transform=transform)
    cloud_trainloader = torch.utils.data.DataLoader(cloud_train,
                                                    batch_size=4, shuffle=True)
    cloud_test = MaskDataset(image_dir, target_image_dir, idx=test_idx,
                             transform=transform, target_transform=transform)
    cloud_testloader = torch.utils.data.DataLoader(cloud_test,
                                                   batch_size=4, shuffle=True)

    return cloud_trainloader, cloud_testloader
