from quicktorch.models import SmallNet
from quicktorch.utils import train, imshow
from quicktorch.data import cifar
import torch

trainloader, testloader, classes = cifar(batch_size=256)

smallnet = SmallNet()
smallnet = smallnet.cuda()

train(smallnet, trainloader)