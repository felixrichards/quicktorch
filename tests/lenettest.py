from quicktorch.models import LeNet
from quicktorch.utils import train
from quicktorch.data import mnist
import torch

trainloader, testloader, classes = mnist(batch_size=256)

lenet = LeNet()
lenet = lenet.cuda()

a, e, p, r = train(lenet, trainloader, device=torch.cuda.current_device())
print(a, e, p, r)
