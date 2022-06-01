import torch
import torch.nn as nn


class ConsensusLoss(nn.Module):
    def __init__(self, eta=.4, lambd=1.1, beta=None, pos_weight=None,
                 seg_criterion=torch.nn.BCEWithLogitsLoss(reduction='none')):
        super().__init__()
        self.eta = eta
        self.beta = beta
        self.lambd = lambd
        self.seg_criterion = seg_criterion

    def forward(self, output, target):
        weight = self.loss_weight(target)
        target = torch.where(target >= self.eta, 1., 0.)
        return (self.seg_criterion(output, target) * weight).mean()

    def loss_weight(self, target):
        p = torch.sum(target >= self.eta)
        n = torch.sum(target == 0)
        weight = torch.zeros_like(target)
        weight[target >= self.eta] = n / (p + n)
        weight[target == 0] = self.lambd * p / (p + n)
        return weight


class PlainConsensusLossMC(ConsensusLoss):
    def __init__(self, eta=.49, **kwargs):
        super().__init__(eta=eta, **kwargs)

    def loss_weight(self, target):
        weight = torch.tensor([1], device=target.device)[None, :, None, None]
        weight = weight.expand_as(target) * (target >= 0)

        return weight


class ConsensusLossMC(ConsensusLoss):
    def __init__(self, eta=.49, **kwargs):
        super().__init__(eta=eta, **kwargs)

    def loss_weight(self, target):
        weight = torch.tensor([1], device=target.device)[None, :, None, None]
        weight = weight.expand_as(target) * torch.logical_or(target == 0., target > 0.49)

        return weight


class SuperMajorityConsensusLossMC(ConsensusLoss):
    def __init__(self, eta=.49, sm=.74, beta=1.25, **kwargs):
        super().__init__(eta=eta, beta=beta, **kwargs)
        self.sm = sm

    def loss_weight(self, target):
        super_maj_weight = torch.tensor([self.beta], device=target.device)[None, :, None, None]
        maj_weight = torch.tensor([1], device=target.device)[None, :, None, None]
        super_maj_weight = super_maj_weight.expand_as(target) * (target >= self.sm)
        maj_weight = maj_weight.expand_as(target) * (target <= 0.24 + torch.logical_and(target >= self.eta, target < self.sm))

        weight = super_maj_weight + maj_weight
        return weight
