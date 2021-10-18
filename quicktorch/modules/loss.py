import torch
import torch.nn as nn


class ConsensusLoss(nn.Module):
    def __init__(self, eta=.4, lambd=1.1, beta=None):
        super().__init__()
        self.eta = eta
        self.beta = beta
        self.lambd = lambd
        self.seg_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, output, target):
        target[target >= self.eta] = 1.
        weight = self.loss_weight(target)
        return (self.seg_criterion(output, target) * weight).mean()  

    def loss_weight(self, target):
        p = torch.sum(target >= self.eta)
        n = torch.sum(target == 0)
        weight = torch.zeros_like(target)
        weight[target >= self.eta] = n / (p + n)
        weight[target == 0] = self.lambd * p / (p + n)
        return weight


class ConsensusLossMC(ConsensusLoss):
    def __init__(self, eta=.4, lambd=1.1, beta=None):
        super().__init__(eta=eta, lambd=lambd, beta=beta)

    def loss_weight(self, target):
        p = torch.sum(target >= self.eta, dim=(0, 2, 3))
        n = torch.sum(target == 0, dim=(0, 2, 3))
        pos_weight = (n / (p + n))[None, :, None, None]
        neg_weight = (self.lambd * p / (p + n))[None, :, None, None]
        pos_weight = pos_weight.expand_as(target) * (target >= self.eta)
        neg_weight = neg_weight.expand_as(target) * (target == 0)

        # if self.beta is not None:
        #     w = torch.sum(0 < target < self.beta, dim=(0, 2, 3))
        #     weak_weight = (self.lambd * p / (p + n))[None, :, None, None]
        #     weak_weight = weak_weight.expand_as(target) * (target == 0)

        weight = pos_weight + neg_weight
        return weight
