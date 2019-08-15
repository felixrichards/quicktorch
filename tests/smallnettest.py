from quicktorch.models import SmallNet
from quicktorch.utils import train, imshow
from quicktorch.data import cifar
import torch

trainloader, testloader, classes = cifar(batch_size=256)

smallnet = SmallNet()
smallnet = smallnet.cuda()

a, e, p, r = train(smallnet, trainloader, device=torch.cuda.current_device())
print(a, e, p, r)