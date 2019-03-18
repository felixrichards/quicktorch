import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from skimage import io


def training_stuff(net, data, opt, criterion, use_cuda, train=True):
    images, labels = data
    # labels = torch.tensor(labels, dtype=torch.float)
    # labels = make_categorical(labels)
    if use_cuda:
        images, labels = images.cuda(), labels.cuda()
    opt.zero_grad()
    with torch.set_grad_enabled(train):
        output = net(images)
        corr = (output*labels).ceil().detach()
        loss = criterion(output, labels)
    if train:
        loss.backward()
        opt.step()
    return loss, corr


def match_shape(out, lbls):
    if out.size(0) != lbls.size(0):
        lbls.unsqueeze_(0)
    return lbls


def train(net, input, criterion='default',
          epochs=5, opt='default', sch=None,
          use_cuda=True,
          save=True, save_best=False, save_all=False):
    # Put model in training mode
    net.train()

    # Record time
    since = time.time()
    best_acc = 0.
    
    # Validate given opt and criterion args
    opt, criterion = validate_opt_crit(opt, criterion, net.parameters())
    
    # Check if a validition input set is provided
    # Store input sizes and batch sizes
    size = {}
    b_size = {}
    if input is list:
        if len(input) == 2:
            phases = ['train', 'val']
            size['train'] = len(input[0])*input[0].batch_size
            size['val'] = len(input[1])*input[1].batch_size
            b_size['train'] = input[0].batch_size
            b_size['val'] = input[1].batch_size
        else:
            print("Invalid data format")
            return
    elif isinstance(input, torch.utils.data.DataLoader):
        phases = ['train']
        size['train'] = len(input)*input.batch_size
        b_size['train'] = input.batch_size

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs))
        print('-' * 15)

        running_loss = 0.0
        running_corr = 0
        # Loop through phases e.g. ['train', 'val']
        for phase in phases:
            if sch is not None:
                if phase == 'train':
                    sch.step()
                    net.train()
                else:
                    net.eval()
            iter_start = time.time()
            avg_time = 0

            # Print progress every 10th of batch size
            print_iter = int(size[phase]/b_size['train']/10)

            for i, data in enumerate(input, 0):
                # Run training process
                loss, corr = training_stuff(net, data, opt,
                                            criterion, use_cuda,
                                            phase == 'train')

                # Increment loss and correct predictions
                running_loss += loss.item()
                running_corr += corr.sum()

                # Update avg iteration time
                iter_end = time.time()
                avg_time = (avg_time*i+iter_end-iter_start)/(i+1)
                iter_start = iter_end

                # Print progress
                if i % print_iter == print_iter - 1:
                    print('Epoch [{}/{}]. Iter [{}/{}]. Loss: {:.4f}. Acc: {:.4f}. Avg time/iter: {:.4f}'
                          .format(
                            epoch+1, epochs, i, size[phase]//b_size['train'],
                            running_loss/(i*b_size[phase]),
                            running_corr.item()/(i*b_size[phase]),
                            avg_time))

            epoch_loss = running_loss/size[phase]
            epoch_acc = running_corr.item()/size[phase]
            print('Epoch {} complete. {} Loss: {:.4f} Acc: {:.4f}'.format(
                  epoch+1, phase, epoch_loss, epoch_acc))

            checkpoint = {
                'epoch': epoch+1,
                'epoch_acc': epoch_acc,
                'optimizer_state_dict': opt.state_dict(),
                'loss': loss
            }

            if save_all:
                net.save(checkpoint=checkpoint)
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                if save_best and not save_all:
                    net.save(checkpoint=checkpoint)


    # Put model in evaluation mode
    net.eval()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


def train_gan(netG, netD, input, criterion='default',
          epochs=1000, optG='default', optD='default', sch=None,
          use_cuda=True,
          save=True, save_best=False, save_all=False):
    # Put model in training mode
    netG.train()
    netD.train()

    # Record time
    since = time.time()
    best_acc = 0.
    
    optG, optD, criterion = validate_opt_crit((optG, optD),
                                        criterion,
                                        (netG.parameters(), netD.parameters()))

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
            ## Train with all-real batch
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

            ## Train with all-fake batch
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
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optG.step()
            del(errG)

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, epochs, i, len(input),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((i == len(input)-1)): #(epoch == epochs-1) and 
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(torchvision.utils.make_grid(fake, padding=2, normalize=True))
                io.imsave("gentests/"+str(epoch)+".png", img_list[-1].permute(1, 2, 0))

            iters += 1
    # for img in img_list:
    #     imshow(img)




def validate_opt_crit(opt, criterion, params):
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
            if opt[i] is 'default':
                opt[i] = optim.SGD(params[i], lr=0.01)
            else:
                raise ValueError('Invalid input for opt')
    opt = tuple(opt)
    if not isinstance(criterion, nn.modules.loss._Loss):
        if criterion is 'default':
            criterion = nn.MSELoss()
        elif criterion is 'bce':
            criterion = nn.BCELoss()
        else:
            raise ValueError('Invalid input for criterion')
    
    return (*opt, criterion)


def make_categorical(labels, classes=10):
    n_labels = torch.zeros(len(labels), classes)
    for i, label in enumerate(labels):
        n_labels[i, int(label)] = 1
    return n_labels


def imshow(img, lbls=None, classes=None):
    force_cpu(img, lbls, classes)

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
                if max(lbls) > 1:
                    lbls = list(lbls.argmax(dim=1))
                else:
                    lbls = list(lbls)
        if type(lbls) is list:
            plt.title(' '.join('%5s' % str(lbls[j]) for j in range(len(lbls))))

    # Show class labels if available
    if classes is not None:
        plt.title(' '.join('%5s' % classes[int(lbls[j])] for j in range(len(lbls))))

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def get_splits(N, n_splits=5):
    kfold = KFold(n_splits=n_splits, shuffle=True)
    splits = kfold.split(range(N))
    splits = np.array(list(splits))

    return splits


def force_cpu(*tensors):
    for t in tensors:
        if isinstance(t, torch.Tensor):
            t.cpu()