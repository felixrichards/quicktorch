from torchvision import transforms
from quicktorch.models import LeNet
from quicktorch.data import mnistrot
from quicktorch.utils import train, evaluate, imshow


def main():
    net = LeNet()
    trainloader, validloader, _ = mnistrot()
    imgs, lbls = next(iter(trainloader))
    print(imgs.max(), imgs.min(), imgs.mean())
    imshow(imgs)
    
    train(net, [trainloader, validloader], epochs=2)



    net = LeNet()
    transform = transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.8, 1.2))
    trainloader, validloader, _ = mnistrot(transform=transform)
    imgs, lbls = next(iter(trainloader))
    print(imgs.max(), imgs.min(), imgs.mean())
    imshow(imgs)

    train(net, [trainloader, validloader], epochs=2)

    # train(net, [trainloader, validloader])
    # testloader, _ = mnistrot(test=True)
    # evaluate(net, testloader)


if __name__ == '__main__':
    main()

