import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import time
import matplotlib.pyplot as plt
import numpy as np
from .vis import TrainPlot
from sklearn.model_selection import KFold
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
    start = time.time()
    images, labels = data
    images, labels = images.to(device), labels.to(device)
    opt.zero_grad()
    with torch.set_grad_enabled(train):
        output = net(images)
        loss = criterion(output, labels)
    # print("Forward pass done in", time.time() - start)
    start = time.time()
    if train:
        loss.backward()
        opt.step()
        # print("Backward pass done in", time.time() - start)
    return loss, output


def train(net, input, criterion='default',
          epochs=5, opt='default', sch=None,
          device="cpu",
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
        else:
            print("Invalid data format")
            return
    elif isinstance(input, torch.utils.data.DataLoader):
        phases = ['train']
        size['train'] = len(input.dataset)
        b_size['train'] = input.batch_size
        input = [input]

    # Get number of classes
    N = 0
    if hasattr(input[0].dataset, 'num_classes'):
        N = input[0].dataset.num_classes
    else:
        N = len(input[0].dataset[0][1])

    # Record time
    since = time.time()
    best_accuracy = 0.
    best_precision = 0.
    best_recall = 0.
    best_epoch = 0
    best_checkpoint = {}

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs))
        print('-' * 15)

        # Loop through phases e.g. ['train', 'val']
        for j, phase in enumerate(phases):
            running_loss = 0.0
            if N is not 0:
                confusion = torch.zeros(N, N)
            else:
                confusion = None
            if sch is not None:
                if phase == 'train':
                    sch.step()
                    net.train()
                else:
                    net.eval()
            iter_start = time.time()
            avg_time = 0

            # Print progress every 10th of batch size
            print_iter = int(size[phase] / b_size['train'] / 10)
            if print_iter == 0:
                print_iter += 1

            for i, data in enumerate(input[j], 0):
                # Run training process
                start = time.time()
                loss, output = perform_pass(net, data, opt,
                                            criterion, device,
                                            phase == 'train')
                # print("Full pass done in", time.time() - start)
                start = time.time()

                if data[0].size() == data[1].size():
                    if phase == 'val':
                        accuracy = 10 * log10(1 / loss.item())
                else:
                    out_idx = output.max(dim=1)[1]
                    lbl_idx = data[1].max(dim=1)[1]
                    if confusion is not None:
                        for j, k in zip(out_idx, lbl_idx):
                            confusion[j, k] += 1
                    corr = confusion.diag()
                    accuracy = corr.sum() / confusion.sum()
                    precision = (corr / confusion.sum(1)).mean()
                    recall = (corr / confusion.sum(0)).mean()

                # Update avg iteration time
                iter_end = time.time()
                avg_time = (avg_time*i+iter_end-iter_start)/(i+1)
                iter_start = iter_end

                # Print progress
                if i % print_iter == print_iter - 1:
                    print('Epoch [{}/{}]. Iter [{}/{}]. Loss: {:.4f}. '
                          'Acc: {:.4f}. '
                          'Precision: {:.4f}. '
                          'Recall: {:.4f}. '
                          'Avg time/iter: {:.4f}. '
                          .format(
                            epoch+1, epochs, i, size[phase]//b_size['train'],
                            running_loss/((i+1)*b_size[phase]),
                            accuracy,
                            precision,
                            recall,
                            avg_time))

            epoch_loss = running_loss/size[phase]
            print('Epoch {} complete. Phase: {}. '
                  'Loss: {:.4f}. '
                  'Acc: {:.4f}. '
                  'Precision: {:.4f}. '
                  'Recall: {:.4f}. '
                  .format(
                    epoch+1, phase, epoch_loss, accuracy,
                    precision, recall))

            checkpoint = {
                'epoch': epoch+1,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'optimizer_state_dict': opt.state_dict(),
                'loss': loss
            }

            if save_all:
                net.save(checkpoint=checkpoint)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_precision = precision
                best_recall = recall
                best_epoch = epoch + 1
                if save_best and not save_all:
                    best_checkpoint = checkpoint

    # Put model in evaluation mode
    net.eval()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    if best_accuracy > 0:
        print('Best accuracy was {} at epoch {}'.format(
            best_accuracy, best_epoch))
        if save_best and not save_all:
            net.save(checkpoint=best_checkpoint)
        return (best_accuracy.item(), best_epoch,
                best_precision.item(), best_recall.item())


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
            else:
                raise ValueError('Invalid input for opt')
    opt = tuple(opt)
    if not isinstance(criterion, nn.modules.loss._Loss):
        if criterion == 'default':
            criterion = nn.MSELoss()
        elif criterion == 'bce':
            criterion = nn.BCELoss()
        else:
            raise ValueError('Invalid input for criterion')

    return (*opt, criterion)


def imshow(img, lbls=None, classes=None):
    """Plots image(s) and their labels with pyplot.

    This function is capable of plotting single or multiple images,
    with class (categorical or not) or mask labels. Will attempt to use
    actual class labels if given.

    Args:
        img (torch.Tensor or PIL.Image.Image): Image data.
        lbls (int or list of int or torch.Tensor): Label data.
        classes (list of str): Class label titles.
    """
    force_cpu(img, lbls, classes)

    if isinstance(img, PIL.Image.Image):
        img = torchvision.transforms.transforms.ToTensor()(img)

    orig_img_size = img.size()
    img = torchvision.utils.make_grid(img)
    if img.min() < 0:
        img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if lbls is not None:
        if isinstance(lbls, torch.Tensor):
            # Labels are masks
            if lbls.size() == orig_img_size:
                lbls = torchvision.utils.make_grid(lbls)
                if lbls.min() < 0:
                    lbls = lbls / 2 + 0.5     # unnormalize
                nplbls = lbls.numpy()
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
            t.cpu()


def main():
    tst_plot = TrainPlot("Loss")
    tst_plot.update_plot(1, 0.5)
    tst_plot.update_plot(2, 0.4)
    tst_plot.update_plot(3, 0.1)


if __name__ == "__main__":
    main()
