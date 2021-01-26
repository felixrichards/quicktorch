import torch
import urllib
from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from quicktorch.utils import train, evaluate


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


urllib.request.urlretrieve(
    'https://lh3.googleusercontent.com/-ELUnFgFJqUU/XPPXOOmhfMI/AAAAAAAAAP0/2cabsTI9uGUYxM3O3w4EOxjR_iJvEQAvACK8BGAs/s374/index3.png',
    'bike.jpg'
)
urllib.request.urlretrieve(
    'https://lh3.googleusercontent.com/-gdUavPeOxdg/XPPXQngAnvI/AAAAAAAAAQA/yoksBterCGQGt-lv3aX4kfyMUDXTar7yACK8BGAs/s374/index4.png',
    'bikemask.jpg'
)
img = ToTensor()(Image.open('./bike.jpg').convert('RGB')).unsqueeze(0)
mask = ToTensor()(Image.open('./bikemask.jpg').convert('L')).unsqueeze(0)
mask[mask > 0] = 1
img = img.repeat(2, 1, 1, 1)
mask = mask.repeat(2, 1, 1, 1)
dset = torch.utils.data.TensorDataset(img, mask)
loader = DataLoader(dset, batch_size=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = Net().to(device)
train(net, [loader, loader], epochs=5, device=device)
evaluate(net, loader, device=device)
