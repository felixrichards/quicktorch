import torch
import torch.nn as nn

from quicktorch.modules.loss import ConsensusLossMC


class DAFLoss(nn.Module):
    def __init__(self,
                 seg_criterion=nn.BCEWithLogitsLoss(),
                 vec_criterion=nn.MSELoss(),
                 pos_weight=7):
        super().__init__()
        self.seg_criterion = seg_criterion
        self.seg_criterion.pos_weight = torch.tensor(pos_weight)
        self.vec_criterion = vec_criterion

    def forward(self, output, target):
        if not (type(output[0]) == tuple or type(output[0]) == list):
            return self.seg_criterion(output, target)
        (
            first_attention_vectors,
            second_attention_vectors,
            attention_in_encodings,
            attention_out_encodings,
            segmentations
        ) = output
        guided_losses = [
            self.vec_criterion(pre_a, a)
            for pre_a, a in zip(first_attention_vectors, second_attention_vectors)
        ]
        reconstruction_losses = [
            self.vec_criterion(in_enc, out_enc)
            for in_enc, out_enc in zip(attention_in_encodings, attention_out_encodings)
        ]
        seg_losses = [
            self.seg_criterion(seg, target)
            for seg in segmentations
        ]
        return sum(seg_losses) + .25 * sum(guided_losses) + .1 * sum(reconstruction_losses)


class DAFLossMCML(ConsensusLossMC):
    """Loss class for DAF network on multiclass multilabel

    Args:
        eta (float, optional): consensus majority value. Defaults to 0.4 to
            compensate for FP errors.
        lambda (float, optional): boosts negative class weighting. Defaults to 1.1.
        seg_criterion (nn._loss, optional): loss fn used for segmentation
            labels.
        vec_criterion (nn._loss, optional): loss fn used for attention
            vectors in self-guided protocol.
    """
    def __init__(self, eta=.4, lambd=1.1, beta=None,
                 seg_criterion=nn.BCEWithLogitsLoss(reduction='none'),
                 vec_criterion=nn.MSELoss()):
        super().__init__(eta=eta, lambd=lambd, beta=beta)
        self.seg_criterion = seg_criterion
        self.vec_criterion = vec_criterion

    def forward(self, output, target):
        target[target >= self.eta] = 1.
        weight = self.loss_weight(target)

        if not (type(output[0]) == tuple or type(output[0]) == list):
            return (self.seg_criterion(output, target) * weight).mean()
        (
            first_attention_vectors,
            second_attention_vectors,
            attention_in_encodings,
            attention_out_encodings,
            segmentations
        ) = output
        # print([seg.mean(dim=(0, 2, 3)) for seg in segmentations])
        # print(target.mean(dim=(0, 2, 3)))
        guided_losses = [
            self.vec_criterion(pre_a, a)
            for pre_a, a in zip(first_attention_vectors, second_attention_vectors)
        ]
        reconstruction_losses = [
            self.vec_criterion(in_enc, out_enc)
            for in_enc, out_enc in zip(attention_in_encodings, attention_out_encodings)
        ]
        seg_losses = [
            (self.seg_criterion(seg, target) * weight).mean()
            for seg in segmentations
        ]
        return sum(seg_losses) + .25 * sum(guided_losses) + .1 * sum(reconstruction_losses)
