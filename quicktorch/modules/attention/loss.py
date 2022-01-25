import torch
import torch.nn as nn
import torch.nn.functional as F

from quicktorch.modules.loss import ConsensusLossMC


class GuidedAuxLoss(nn.Module):
    def __init__(self, criterion=nn.MSELoss()):
        super().__init__()
        self.criterion = criterion

    def forward(self, aux_outputs):
        guided_losses, reconstruction_losses = zip(*[self.aux_guided_loss(aux) for aux in aux_outputs])
        return .25 * sum(guided_losses) + .1 * sum(reconstruction_losses)

    def aux_guided_loss(self, aux):
        return (
            self.criterion(aux['in_semantic_vectors'], aux['out_semantic_vectors']),
            self.criterion(aux['in_attention_encodings'], aux['out_attention_encodings'])
        )


class DAFLoss(nn.Module):
    def __init__(self,
                 seg_criterion=nn.BCEWithLogitsLoss(),
                 aux_loss=GuidedAuxLoss(),
                 pos_weight=None):
        super().__init__()
        self.seg_criterion = seg_criterion
        if pos_weight is not None:
            self.seg_criterion.pos_weight = pos_weight
        self.aux_loss = aux_loss

    def forward(self, output, target):
        if not (type(output[0]) == tuple or type(output[0]) == list):
            return self.seg_criterion(output, target)

        segmentations, aux_outputs = output

        seg_losses = [
            self.seg_criterion(seg, target)
            for seg in segmentations
        ]

        if all(o is None for o in aux_outputs):
            return sum(seg_losses)

        aux_loss = self.aux_loss(aux_outputs)
        return sum(seg_losses) + aux_loss


class DAFConsensusLoss(DAFLoss):
    """Loss class for DAF network on multiclass multilabel

    Args:
        seg_criterion (nn._loss, optional): loss fn used for segmentation
            labels.
        aux_loss (nn._loss, optional): loss fn used for attention
            vectors in self-guided protocol.
    """
    def __init__(self,
                 consensus_criterion=ConsensusLossMC(eta=.49, lambd=1.1, beta=None),
                 aux_loss=GuidedAuxLoss(),
                 pos_weight=None):
        super().__init__(
            seg_criterion=consensus_criterion,
            aux_loss=aux_loss,
            pos_weight=pos_weight
        )


class FocalWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=2., gamma=2., reduction='none'):
        super().__init__()
        self.alpha = 1 / (1 + 1 / pos_weight)
        self.gamma = gamma
        self.eps = 1e-6
        self.reduction = reduction

    def forward(self, output, target):
        prob = torch.sigmoid(output).clip(self.eps, 1. - self.eps)

        # not perfect, consider reimplementing with log_softmax for numerical stability
        back_ce = -(1 - target) * torch.pow(prob, self.gamma) * torch.log(1 - prob)
        fore_ce = -target * torch.pow(1 - prob, self.gamma) * torch.log(prob)

        out = (1 - self.alpha) * back_ce + self.alpha * fore_ce
        if self.reduction == 'none':
            return out
        return out.mean()


class ClassBalancedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None, beta=.9999, reduction='none'):
        super().__init__()
        self.pos_weight = pos_weight
        self.n_classes = pos_weight.shape[0]
        self.beta = beta
        self.reduction = reduction

    def forward(self, output, target):
        """Compute the Class Balanced Loss between `output` and the ground truth `target`.

        Class Balanced Loss: ((1 - beta) / (1 - beta ^ n)) * L(target, output)
        where L is one of the standard losses used for Neural Networks.
        Args:
        output: A float tensor of size [batch, self.n_classes].
        target: A int tensor of size [batch, self.n_classes].
        samples_per_cls: A python list of size [self.n_classes].
        self.n_classes: total number of classes. int
        loss_type: string. One of "sigmoid", "focal", "softmax".
        beta: float. Hyperparameter for Class balanced loss.
        Returns:
        cb_loss: A float tensor representing class balanced loss
        """
        effective_num = 1.0 - torch.pow(self.beta, self.pos_weight)
        weights = (1.0 - self.beta) / effective_num
        weights = weights / torch.sum(weights) * self.n_classes  # [c]

        weights = weights.view(1, self.n_classes, 1, 1) # [1, c] [1, c, w, h]
        weights = weights.repeat(target.shape[0], 1, 1, 1) * target  # [n, c] * [n, c]  [n, c, w, h]
        weights = weights.sum(1, keepdim=True) # [n, 1, w, h]
        weights = weights.repeat(1, self.n_classes, 1, 1) # [n, c, w, h]

        return F.binary_cross_entropy_with_logits(input=output, target=target, weight=weights, reduction=self.reduction)


class AsymmetricFocalTvesrkyWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None, delta=0.6, gamma=0.2, reduction='none'):
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-8

    def forward(self, output, target):
        # Clip values to prevent division by zero error
        output = torch.sigmoid(output).clip(self.eps, 1. - self.eps)

        # Calculate true positives (tp), false negatives (fn) and false positives (fp)     
        axis = (-1, -2)
        tp = torch.sum(target * output, axis=axis)
        tn = torch.sum((1 - target) * (1 - output), axis=axis)
        fn = torch.sum(target * (1 - output), axis=axis)
        fp = torch.sum((1 - target) * output, axis=axis)
        dice_class_pos = (tp + self.eps) / (tp + self.delta * fn + (1 - self.delta) * fp + self.eps)
        dice_class_neg = (tn + self.eps) / (tn + self.delta * fp + (1 - self.delta) * fn + self.eps)
        # Calculate losses separately for each class, only enhancing foreground class
        back_dice = (1 - dice_class_neg)
        fore_dice = (1 - dice_class_pos) * torch.pow(1 - dice_class_pos, -self.gamma)

        loss = back_dice + fore_dice

        return loss.unsqueeze(-1).unsqueeze(-1)


class UnifiedFocalWithLogitsLoss(nn.Module):
    def __init__(self, weight=0.5, pos_weight=None, delta=0.6, gamma=0.2, reduction='none'):
        super().__init__()
        self.weight = weight
        self.delta = delta
        self.gamma = gamma
        self.reduction = reduction
        self.aft_loss = AsymmetricFocalTvesrkyWithLogitsLoss(
            delta=delta,
            gamma=gamma,
            reduction=reduction
        )
        self.eps = 1e-8

    def forward(self, output, target):
        return (
            self.weight * self.aft_loss(output, target)
            # ((1 - self.weight) * self.asymmetric_focal_loss(output, target))
        )

    def asymmetric_focal_loss(self, output, target):
        # output = torch.sigmoid(output).clip(self.eps, 1. - self.eps)

        # calculate losses separately for each class, only suppressing background class
        back_ce = -(1 - target) * torch.pow(1 - torch.sigmoid(output).clip(self.eps, 1. - self.eps), self.gamma) * torch.log(1 - torch.sigmoid(output).clip(self.eps, 1. - self.eps))

        fore_ce = -target * F.logsigmoid(output)

        # focal_factor = (1 - target).mul_(torch.pow(1 - torch.sigmoid(output).clip(self.eps, 1. - self.eps), self.gamma))
        # loss = (output * focal_factor).add_((-output).exp() + 1).mul_()
        # loss = (-output).exp_().add_(1).log_().mul_(focal_factor + target).add_(x * focal_factor)

        # loss = (1 - target).mul_((1 - output).pow_(self.gamma)).mul_(torch.log(torch.sigmoid(output).mul_(-1).add_(1))).add_(target.mul_(F.logsigmoid(output)))

        return (1 - self.delta) * back_ce + self.delta * fore_ce
