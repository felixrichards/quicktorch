from quicktorch.models import LeNet
from quicktorch.data import mnistrot
from quicktorch.utils import train, evaluate


def main():
    net = LeNet()
    trainloader, validloader, _ = mnistrot()
    train(net, [trainloader, validloader])

    testloader, _ = mnistrot(test=True)
    evaluate(net, testloader)


if __name__ == '__main__':
    main()

