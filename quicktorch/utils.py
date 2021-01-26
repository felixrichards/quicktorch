import os
from urllib.parse import urlparse
from urllib.request import urlopen
import zipfile
import tarfile
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from .metrics import MetricTracker
from .vis import TrainPlot
from sklearn.model_selection import KFold
from sklearn.metrics import jaccard_score
from skimage import io
import PIL
from math import log10
"""This module provides miscellaneous helper functions for neural network
experimentation, most notably a training function.
"""


def perform_pass(net, data, opt, criterion, device, train=True):
    """Performs a forward and backward pass.

    Args:
        net (torch.nn.Module): Network to propogate.
        data (list of torch.Tensor): List containing [images, labels].
        opt (torch.optim.Optimizer): Optimizer function.
        criterion (torch.nn.modules.loss._Loss): Criterion function.
        device (str): Index of GPU or 'cpu' if no GPU.
        train (boolean, optional): Whether to perform backward prop.
            Defaults to True.

    Returns:
        float: Loss value
        float: Number of correct predictions
    """
    images, labels = data
    images, labels = images.to(device), labels.to(device)
    opt.zero_grad()
    with torch.set_grad_enabled(train):
        output = net(images)
        loss = criterion(output, labels)
    if train:
        loss.backward()
        opt.step()
    return loss, output


def _handle_sch(sch, running_loss=None, phase='train'):
    if phase == 'train':
        if sch is not None:
            if isinstance(sch, optim.lr_scheduler.ReduceLROnPlateau):
                sch.step(running_loss)
            else:
                sch.step()


def train(net, input, criterion='default',
          epochs=5, opt='default', sch=None,
          metrics=None, device="cpu",
          save_best=False, save_all=False):
    """Trains a neural network

    This function was written to use extra features added
    through the quicktorch.Model wrapper, though is still compatible
    with a naked net.

    Args:
        net (torch.nn.Module): Network to be trained.
        input (torch.utils.data.dataloader.Dataloader or
            list of torch.utils.data.dataloader.Dataloader): Training data.
            If 2 dataloaders are passed, the 2nd will be used for validation.
        criterion (torch.nn.modules.loss._Loss, optional): Loss function.
            Defaults to 'default' which translates to MSE.
        epochs (int, optional): Number of epochs to train over.
            Defaults to 5.
        opt (torch.optim.Optimizer, optional): Optimiser function.
            Defaults to 'default' which translates to SGD.
        sch (torch.optim._LRScheduler, optional): Learning rate scheduling
            function. Defaults to None.
        metrics (quicktorch.metrics.MetricTracker, optional): Class for tracking
            metrics. If None attempts to detect suitable metrics from data.
            Defaults to None.
        device (str, optional): Index of GPU or 'cpu' if no GPU.
            Defaults to 'cpu'.
        save_best (boolean, optional): Saves the model at the best epoch.
            Defaults to False.
        save_all (boolean, optional): Saves the model at all epochs.
            Defaults to False.

    Returns:
        float: Best accuracy of model.
        int: Epoch which had best accuracy.
        float: Precision of model at best epoch.
        float: Recall of model at best epoch.
    """
    # Put model in training mode
    net.train()

    # Validate given opt and criterion args
    opt, criterion = _validate_opt_crit(opt, criterion, net.parameters())

    # Check if a validition input set is provided
    # Store input sizes and batch sizes
    size = {}
    b_size = {}
    if type(input) is list:
        if len(input) == 2:
            phases = ['train', 'val']
            size['train'] = len(input[0].dataset)
            size['val'] = len(input[1].dataset)
            b_size['train'] = input[0].batch_size
            b_size['val'] = input[1].batch_size
            # Create temp file for storing best model dict
            temp_model_file = tempfile.NamedTemporaryFile(mode='wb', delete=False)
            temp_model_file.close()
        else:
            print("Invalid data format")
            return
    elif isinstance(input, torch.utils.data.DataLoader):
        phases = ['train']
        size['train'] = len(input.dataset)
        b_size['train'] = input.batch_size
        input = [input]

    if metrics is None:
        metrics = MetricTracker.detect_metrics(input)

    metrics.start()
    best_checkpoint = {}

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs))
        print('-' * 15)

        # Loop through phases e.g. ['train', 'val']
        for j, phase in enumerate(phases):
            if phase == 'train':
                net.train()
            else:
                net.eval()
            print(f'phase={phase}, net.training={net.training}')
            running_loss = 0.0

            # Print progress every 10th of batch size
            print_iter = int(size[phase] / b_size[phase] / 10)
            if print_iter == 0:
                print_iter += 1

            for i, data in enumerate(input[j], 0):
                # Run training process
                loss, output = perform_pass(net, data, opt,
                                            criterion, device,
                                            phase == 'train')
                running_loss += loss.item()

                metrics.update(output, data[1].to(device))
                del(output)
                del(loss)

                # Print progress
                if i % print_iter == print_iter - 1:
                    print('Epoch [{}/{}]. Iter [{}/{}]. Loss: {:.4f}. '
                          '{} '
                          '{} '
                          .format(
                            epoch+1, epochs, i, size[phase]//b_size[phase],
                            running_loss/((i+1)*b_size[phase]),
                            metrics.progress_str(),
                            metrics.stats_str()))

            epoch_loss = running_loss/size[phase]
            print('Epoch {} complete. Phase: {}. '
                  'Loss: {:.4f}. '
                  '{} '
                  '{} '
                  .format(
                    epoch+1, phase, epoch_loss,
                    metrics.progress_str(),
                    metrics.stats_str()))

            if len(phases) == 1 or phase == 'val':
                checkpoint = {
                    'epoch': epoch+1,
                    'optimizer_state_dict': opt.state_dict(),
                    'loss': epoch_loss
                }
                checkpoint.update(metrics.get_metrics())
                if metrics.is_best():
                    best_epoch = epoch + 1
                    torch.save(net.state_dict(), temp_model_file.name)
                    if save_best and not save_all:
                        best_checkpoint = checkpoint
                metrics.show_best()

            if save_all:
                net.save(checkpoint=checkpoint)
            _handle_sch(sch, epoch_loss, phase)
            metrics.reset()

    if 'val' in phases and best_checkpoint is not None:
        net.load_state_dict(torch.load(temp_model_file.name))
        os.remove(temp_model_file.name)
    # Put model in evaluation mode
    metrics.finish()
    net.eval()
    best_metrics = metrics.get_best_metrics()
    if save_best and not save_all:
        save_path = net.save(checkpoint=best_checkpoint)
        best_metrics['save_path'] = save_path
    return best_metrics


def evaluate(net, input, device='cpu', metrics=None):
    """Evaluates a model on a given input
    """
    net.eval()
    size = len(input.dataset)
    b_size = input.batch_size

    if metrics is None:
        metrics = MetricTracker.detect_metrics(input)

    # Print progress every 10th of batch size
    print_iter = int(size / b_size / 10)
    if print_iter == 0:
        print_iter += 1

    for i, data in enumerate(input, 0):
        # Run training process
        with torch.no_grad():
            output = net(data[0].to(device))
        metrics.update(output, data[1].to(device))
        del(output)

        # Print progress
        if i % print_iter == print_iter - 1:
            print('Iter [{}/{}] '
                  '{} '
                  '{} '
                  .format(
                    i, size//b_size,
                    metrics.progress_str(),
                    metrics.stats_str()))
    print('---')
    print('Final results.')
    print('---')
    print(metrics.progress_str())
    return metrics.get_metrics()


def train_gan(netG, netD, input, criterion='default',
              epochs=1000, optG='default', optD='default', sch=None,
              use_cuda=True,
              save=True, save_best=False, save_all=False):
    """Trains a generator and discriminator.

    This function is unfinished and mostly copied from the Pytorch
    tutorials.
    """
    # Put model in training mode
    netG.train()
    netD.train()

    optG, optD, criterion = _validate_opt_crit((optG, optD),
                                               criterion,
                                               (netG.parameters(),
                                                netD.parameters()))

    fixed_noise = torch.randn(64, netG.nz, 1, 1).cuda()

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(epochs):
        # For each batch in the dataloader
        for i, data in enumerate(input, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].cuda()
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), 1).cuda()
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            # Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, netG.nz, 1, 1).cuda()
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(0)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optD.step()
            del(errD_fake)

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(1)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass
            # of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f \
                       \tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, epochs, i, len(input),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            del(errD)
            del(errG)

            # Check on generator by saving G's output on fixed noise
            if (iters % 500 == 0) or ((i == len(input)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img = torchvision.utils.make_grid(fake, padding=2,
                                                  normalize=True)
                img_list.append(img)
                io.imsave("gentests/"+str(epoch)+".png",
                          img_list[-1].permute(1, 2, 0))

            iters += 1


def _validate_opt_crit(opt, criterion, params):
    """Validates optimiser and criterion

    Can pass multiple optimisers/parameters.
    """
    # Set default optimiser to SGD
    if type(opt) is not tuple:
        opt = (opt,)
    if type(params) is not tuple:
        params = (params,)

    if len(opt) != len(params):
        raise ValueError('No of optimizers does not match no of params')

    opt = list(opt)
    for i in range(len(opt)):
        if not isinstance(opt[i], optim.Optimizer):
            if opt[i] == 'default':
                opt[i] = optim.SGD(params[i], lr=0.01)
    opt = tuple(opt)
    if not isinstance(criterion, nn.modules.loss._Loss):
        if criterion == 'default':
            criterion = nn.MSELoss()
        elif criterion == 'bce':
            criterion = nn.BCELoss()

    return (*opt, criterion)


def imshow(img, lbls=None, classes=None, save_name=None):
    """Plots image(s) and their labels with pyplot.

    This function is capable of plotting single or multiple images,
    with class (categorical or not) or mask labels. Will attempt to use
    actual class labels if given.

    Args:
        img (torch.Tensor or PIL.Image.Image): Image data.
        lbls (int or list of int or torch.Tensor): Label data.
        classes (list of str): Class label titles.
        save_name (str): Name to save figure under. If None will not be saved.
    """
    force_cpu(img, lbls, classes)

    if isinstance(img, PIL.Image.Image):
        img = torchvision.transforms.transforms.ToTensor()(img)

    orig_img_size = img.size()
    img = torchvision.utils.make_grid(img)
    if img.min() < 0:
        img = (img - img.min()) / (img.max() - img.min())   # normalize
    npimg = img.cpu().detach().numpy()
    if lbls is not None:
        if isinstance(lbls, torch.Tensor):
            # Labels are masks
            if lbls.size(-1) == orig_img_size[-1] and lbls.size(-2) == orig_img_size[-2]:
                lbls = torchvision.utils.make_grid(lbls)
                if lbls.min() < 0:
                    lbls = (lbls - lbls.min()) / (lbls.max() - lbls.min())     # unnormalize
                nplbls = lbls.cpu().detach().numpy()
                if npimg.shape[-3] == 3 and nplbls.shape[-3] == 1:
                    nplbls = np.repeat(nplbls, 3, -3)
                npimg = np.hstack((npimg, nplbls))
            # Labels are categorical
            else:
                if lbls.dim() > 1:
                    lbls = list(lbls.argmax(dim=1))
                else:
                    lbls = list(lbls)
        if type(lbls) is int:
            assert(lbls.size(0) == 1)
            lbls = list(lbls)
        if type(lbls) is list:
            plt.title(' '.join('%5s' % str(lbls) for lbl in lbls))

    # Show class labels if available
    if classes is not None:
        plt.title(' '.join('%5s' % classes[int(lbl)] for lbl in lbls))

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.axis('tight')
    plt.axis('image')
    if save_name:
        if save_name[len(save_name) - 4:] != '.png':
            save_name += '.png'
        plt.savefig(save_name)
    plt.show()


def get_splits(N, n_splits=5):
    """Produces validation split indices.

    Args:
        N (int): Size of validation dataset.
        n_splits (int, optional): Desired number of splits.
            Defaults to 5.

    Returns:
        np.array: K Fold validation splits.
    """
    kfold = KFold(n_splits=n_splits, shuffle=True)
    splits = kfold.split(range(N))
    splits = np.array(list(splits))

    return splits


def force_cpu(*tensors):
    """Forces input tensors to be on CPU.

    Args:
        *tensors (torch.Tensor): tensors to force to CPU.
    """
    for t in tensors:
        if isinstance(t, torch.Tensor):
            t.cpu().detach()


def download(url, dir, name=None, extract=True):
    urlpath = urlparse(url).path
    dlname = urlpath.split('/')[-1]
    if name is None:
        name = dlname

    if not os.path.isdir(dir):
        os.mkdir(dir)
    save_path = os.path.join(dir, name)
    request = urlopen(url)
    with open(save_path, 'wb') as f:
        f.write(request.read())
    print("Downloaded " + url + " to " + save_path)
    if extract:
        print('Extracting')
        if _get_ext(dlname) == '.zip':
            with zipfile.ZipFile(save_path, "r") as zip_ref:
                zip_ref.extractall(dir)
        if _get_ext(dlname) == '.tgz':
            with tarfile.open(save_path) as tar:
                for item in tar:
                    tar.extract(item, dir)


def _get_ext(s):
    return '.' + s.split('.')[-1]


def main():
    tst_plot = TrainPlot("Loss")
    tst_plot.update_plot(1, 0.5)
    tst_plot.update_plot(2, 0.4)
    tst_plot.update_plot(3, 0.1)


if __name__ == "__main__":
    main()
