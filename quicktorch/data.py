import torch
import torchvision
from torchvision import transforms
from .customtransforms import ConvertType, MakeCategorical
from .datasets import MaskDataset


def cifar(alexnet=False, batch_size=4):
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

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform,
                                            target_transform=target_transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform,
                                           target_transform=target_transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes

def mnist(batch_size=32):

    transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                    ])
    target_transform = MakeCategorical()

    trainloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('../data', train=True, download=True,
                        transform=transform,
                        target_transform=target_transform),
        batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('../data', train=False, transform=transform,
                        target_transform=target_transform),
        batch_size=batch_size, shuffle=True)
    
    classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")

    return trainloader, testloader, classes


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
