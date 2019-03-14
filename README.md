# quicktorch
My simple pytorch wrapper with some utils for quick prototyping.

### Installation

todo


### Features

##### Custom transforms for loading datasets in a convenient format:

```
from quicktorch.customtransforms import ConvertType, MakeCategorical
import torchvision

target_transform = transforms.Compose([
    ConvertType(torch.float),
    MakeCategorical(10)
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform,
                                        target_transform=target_transform)
```

##### Common dataset wrappers:

```
from quicktorch.data import cifar

trainloader, testloader, classes = cifar(alexnet=True, batch_size=256)
```

##### Extensions of Dataset class for common dataset types. 

For classification dataset create a CSV file with headers "imagename,label" and put it in image folder:

```
from quicktorch.datasets import ClassificationDataset
from quicktorch.utils import imshow
import torch

dataset = ClassificationDataset("path/to/file.csv")
print(len(dataset))

dataloader = torch.utils.data.Dataloader(dataset)

images, labels = next(iter(dataloader))
imshow(images)
```

For mask dataset:

```
from quicktorch.datasets import MaskDataset
from quicktorch.utils import imshow
import torch

dataset = MaskDataset("path/to/images/", "path/to/masks/")
print(len(dataset))

dataloader = torch.utils.data.Dataloader(dataset)

images, labels = next(iter(dataloader))
imshow(images)
```

##### Extension of Model class for extra utilities

Enabling pretraining and transfer learning for new dataset with N classes:

```
from quicktorch.models import AlexNet
from quicktorch.datasets import ClassificationDataset

dataset = ClassificationDataset("path/to/file.csv")
dataloader = torch.utils.data.Dataloader(dataset)

alexnet = AlexNet()
alexnet.pretrain()
alexnet.transfer_learn()
alexnet.change_last_fcn(num_classes=dataset.num_classes)
```

Easy saving and loading:

```
from quicktorch.models import SmallNet
from quicktorch.data import cifar
from quicktorch.utils import train

trainloader, _, _ = cifar(alexnet=True, batch_size=256)
net = SmallNet(name="mycnntest")
train(net, trainloader)
net.save()

net = None
net = SmallNet(name="mycnntest")
net.load()

net = None
net = SmallNet(name="mysecondcnntest")
train(net, trainloader, save_best=True)
```

##### Other miscellaneous utilities

Some features:
* Neat image vis for classification and mask datasets (shown above)
* k-Fold splits
* Training functions for standard CNN and GAN (shown above)

```
import quicktorch.utils as utils
from quicktorch.datasets import MaskDataset
import glob

splits = get_splits(len(glob.glob('path/to/images/*.png'))

for train_idx, test_idx in splits:
    dataset = MaskDataset("path/to/images/", "path/to/masks/",
                          train_idx=train_idx, test_idx=test_idx)
    # train network on each fold
```
