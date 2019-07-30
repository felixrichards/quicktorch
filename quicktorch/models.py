import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import os
"""This module provides a nn.Module wrapper and some standard model
architectures.

Attributes:
    model_urls (dict of str): Contains URLs for pretrained weights of
        preset models.
"""

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class Model(nn.Module):
    """Wraps the neural network class.

    Attributes:
        name (str): Simple string identifier for the model. Is used when
            saving the model.

    Args:
        **kwargs:
            weights_url (str, optional): A URL to download pre-trained weights.
            name (str, optional): See above. Defaults to None.
    """
    name = None

    def __init__(self, **kwargs):
        super(Model, self).__init__()
        if "weights_url" in kwargs is None:
            self.weights_url = kwargs.pop("weights_url")
        if "name" in kwargs is None:
            self.name = kwargs.pop("name")

    def change_last_fcn(self, num_classes=10, layer=None):
        """Modifies the last fully connected layer for transfer learning.

        Args:
            num_classes (int, optional): Number of class labels in new dataset.
                Defaults to 10.
            layer (nn.Linear, optional): Layer to replace current with.
                Defaults to None.
        """
        if layer is not None:
            if isinstance(layer, nn.Linear):
                self.classifier.__setitem__(-1, layer)
            else:
                raise ValueError("Invalid layer object given")
        size = self.classifier[-1].in_features
        self.classifier.__setitem__(-1, nn.Linear(size, num_classes))

    def pretrain(self, weights=None, weights_url=None):
        """Sets weights to pretrained values.

        Args:
            weights (dict, optional): Must be in the format of PyTorch's
                state_dict.
            weights_url ('str', optional): URL to download weights from.
                Not required to be passed if set when the class was
                instantiated.
        """
        if hasattr(self, 'weights_url'):
            self.load_state_dict(model_zoo.load_url(self.weights_url))
        else:
            print("No weights to load")

    def num_flat_features(self, x):
        """Computes the number of features per sample in an input.

        Args:
            x (torch.Tensor): Input data.
        """
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def transfer_learn(self):
        """Disables learning on all but the last layer.
        """
        params = [p for p in self.parameters()]
        del(params[-1])
        for p in params:
            p.requires_grad = False

    def load(self, name=None, save_dir="models"):
        """Attemps to load a state of the model.

        This function does not support the loading of additional
        information, e.g. optimizer weights; only model weights
        will be loaded from the state_dict.

        Args:
            name (str, optional): Filename.
            save_dir (str, optional): Directory to look for data.
                Defaults to "models".
        """
        if name is None:
            name = self.name
        save_path = os.path.join(save_dir, name)

        if not (os.path.isfile(save_path) or
                os.path.isfile(save_path+".pk")):
            print("No saved model under", name+".pk")
            return

        # Check if path has extension
        if os.path.isfile(save_path+".pk"):
            save_path += ".pk"

        checkpoint = torch.load(save_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        print("Model successfully loaded")
        del (checkpoint['model_state_dict'])
        if checkpoint:
            print(checkpoint)
            print("File has other items other than model weights:")
            for key in checkpoint:
                print("\t", key)
            print("Consider loading them with torch.load()")

    def save(self, name=None, overwrite=False, save_dir="models",
             checkpoint=None):
        """Saves a state of the model.

        If a checkpoint dict is passed, the epoch number will be appended
        to the filename, e.g. alexnet.pk < alexnet_epoch1.pk.

        Args:
            name (str, optional): Model identifier or filename.
                A full filepath may work but is not supported. Use save_dir.
                Extension is not required.
            overwrite (boolean, optional): If a file matching the given
            save_dir (str, optional): Directory to save model inside.
            checkpoint (dict, optional): Contains additional information about
            the save point. Can also include information to restart training
                from. For example:
                checkpoint = {
                    'epoch': epoch,
                    'optimizer_state_dict': opt.state_dict()
                }
        """
        if name is None:
            if self.name is not None:
                name = self.name
            else:
                print("No filename given and no default filename \
                       exists for this model.")
                print("Please enter a name to save to")
                name = input("Filename: ")
                while name == "":
                    print("Empty string. Try again.")
                    name = input("Filename: ")

        if not os.path.exists(save_dir):
            print("Folder does not exist, would you like to create it?")
            ans = input("[Y/N]: ")
            while not (ans == "Y" or ans == "y" or ans == "N" or ans == "n"):
                print("Invalid input. Try again.")
                ans = input("[Y/N]: ")
            if ans == "Y" or "y":
                os.makedirs(save_dir)
            else:
                print("Aborting.")

        name = name.replace('.pk', '')

        save_path = os.path.join(save_dir, name)
        save_obj = {'model_state_dict': self.state_dict()}
        if checkpoint is not None:
            assert isinstance(checkpoint, dict)
            save_obj.update(checkpoint)
            if "epoch" in save_obj:
                save_path += "_epoch"+str(save_obj["epoch"])

        if not overwrite and (os.path.isfile(save_path) or
                              os.path.isfile(save_path+".pk")):
            print("File exists, adding number to filename prevent overwriting")
            i = 1
            n_save_path = save_path + str(i)
            while (os.path.isfile(n_save_path) or
                   os.path.isfile(n_save_path+".pk")):
                i += 1
                n_save_path = save_path + str(i)
            save_path = n_save_path

        if os.path.splitext(save_path)[-1] != ".pk":
            save_path += ".pk"

        torch.save(save_obj, save_path)
        print("Successfully saved at " + save_path)


class AlexNet(Model):
    """AlexNet replica.

    Attributes:
        weights_url (str): URL to download weights from.
        name (str): Model identifier.

    Args:
        num_classes (int): Number of classes.
        **kwargs: See torch.models.Model.
    """
    name = 'alexnet'
    weights_url = model_urls['alexnet']

    def __init__(self, num_classes, **kwargs):
        super(AlexNet, self).__init__(**kwargs)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


class LeNet(Model):
    """LeNet replica.

    Attributes:
        name (str): Model identifier.

    Args:
        **kwargs: See torch.models.Model.
    """
    name = "lenet"

    def __init__(self, **kwargs):
        super(LeNet, self).__init__(**kwargs)
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SmallNet(Model):
    """A lower parameter space version of AlexNet.

    Attributes:
        name (str): Model identifier.

    Args:
        **kwargs: See torch.models.Model.
    """
    name = "smallnet"

    def __init__(self, **kwargs):
        super(SmallNet, self).__init__(**kwargs)
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 64 * 4 * 4)
        x = self.classifier(x)
        return x


class Generator(Model):
    """Basic Generator network.

    This model is essentially copied from the PyTorch tutorials.

    Attributes:
        name (str): Model identifier.

    Args:
        nf (int, optional): Filter channel multiplier.
            Defaults to 64.
        nc (int, optional): Number of desired channels in output.
            Defaults to 3.
        nz (int, optional): Number of channels in the latent vector z.
            Defaults to 100.
        **kwargs: See torch.models.Model.
    """
    name = "generator_test"

    def __init__(self, nf=64, nc=3, nz=100, **kwargs):
        super(Generator, self).__init__(**kwargs)
        self.nz = nz
        self.generate = nn.Sequential(
            nn.ConvTranspose2d(nz, nf*8, 5, 2, 0, bias=False),
            nn.BatchNorm2d(nf*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(nf*8, nf*4, 5, 2, 0, bias=False),
            nn.BatchNorm2d(nf*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(nf*4, nf*2, 5, 2, 0, bias=False),
            nn.BatchNorm2d(nf*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(nf*2, nf, 5, 2, 0, bias=False),
            nn.BatchNorm2d(nf),
            nn.ReLU(True),
            nn.ConvTranspose2d(nf, nc, 5, 2, 0, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.generate(x)


class Discriminator(Model):
    """Basic Discriminator network.

    This model is essentially copied from the PyTorch tutorials.

    Attributes:
        name (str): Model identifier.

    Args:
        nf (int, optional): Filter channel multiplier.
            Defaults to 64.
        nc (int, optional): Number of desired channels in output.
            Defaults to 3.
        **kwargs: See torch.models.Model.
    """
    name = "discriminator_test"

    def __init__(self, nf=64, nc=3, **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.discriminate = nn.Sequential(
            nn.Conv2d(nc, nf, 5, 2, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf*2, 5, 2, 0, bias=False),
            nn.BatchNorm2d(nf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*2, nf*4, 5, 2, 0, bias=False),
            nn.BatchNorm2d(nf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*4, nf*8, 5, 2, 0, bias=False),
            nn.BatchNorm2d(nf*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*8, 1, 5, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.discriminate(x)


if __name__ == "__main__":
    alexnet = AlexNet()
    alexnet.pretrain()
    # alexnet.change_last_fcn()
    print(alexnet)
    alexnet.save(name="AlexNet_Test", checkpoint={'epoch': 20})
