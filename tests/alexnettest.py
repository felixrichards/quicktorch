from quicktorch.models import AlexNet
from quicktorch.utils import train, imshow
from quicktorch.data import cifar
import torch.optim as optim
from torch.optim import lr_scheduler


trainloader, testloader, classes = cifar(alexnet=True, batch_size=8)

# images, labels = next(iter(trainloader))
# imshow(images, labels, classes)

alexnet = AlexNet()
alexnet.pretrain()
alexnet.change_last_fcn()
alexnet = alexnet.cuda()

# transfer_opt = optim.SGD(alexnet.classifier[-1].parameters(), lr=0.01)
# exp_lr_scheduler = lr_scheduler.StepLR(transfer_opt, step_size=7, gamma=0.1)

train(alexnet, trainloader)
