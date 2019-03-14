from quicktorch.models import SmallNet
from quicktorch.utils import train, imshow
from quicktorch.data import cifar
import torch

trainloader, testloader, classes = cifar(alexnet=True, batch_size=256)

smallnet = SmallNet()
smallnet = smallnet.cuda()

# images, labels = next(iter(trainloader))
# imshow(images, labels, classes)
# testimg = torch.randn(1, 3, 32, 32).cuda()
# testout = smallnet(testimg)

train(smallnet, trainloader)