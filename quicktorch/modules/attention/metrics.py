import torch
from quicktorch.metrics import MultiClassSegmentationTracker, iou, dice
import matplotlib.pyplot as plt


class DAFMetric(MultiClassSegmentationTracker):
    """Tracks metrics for dual attention features networks.
    """
    def __init__(self, full_metrics=False, n_classes=1, eta=.49):
        super().__init__(full_metrics=full_metrics, n_classes=n_classes)
        self.eta = eta

    def calculate(self, output, target):
        """Calculates metrics for given batch
        """
        if type(output[0]) == tuple or type(output[0]) == list:
            segmentations = output[0]
            seg_pred = sum(segmentations) / len(segmentations)
            plot = False
        else:
            seg_pred = output
            plot = False

        target = target.detach()
        target[target >= self.eta] = 1
        target[target < self.eta] = 0
        # target = target.to(torch.int32)
        seg_pred = seg_pred.detach()
        seg_pred = torch.sigmoid(seg_pred)

        if self.n_classes == 1:
            iou_ = self.iou_fn(seg_pred, target)
            dice_ = self.dice_fn(seg_pred, target)
        else:
            iou_ = torch.tensor([
                self.iou_fn(
                    seg_pred[:, i].contiguous(),
                    target[:, i].contiguous()
                ) for i in range(self.n_classes)
            ])
            if not self.full_metrics:
                iou_ = iou_.mean()
            dice_ = self.dice_fn(
                seg_pred.cpu().contiguous().round().numpy(),
                target.cpu().contiguous().numpy()
            )

        self.metrics['IoU'] = self.batch_average(iou_, 'IoU')
        self.metrics["Dice"] = self.batch_average(dice_, 'Dice')

        if plot:
            fig, ax = plt.subplots(2, self.n_classes)
            if seg_pred.ndim == 3:
                ax[0].imshow(target[0], vmin=0, vmax=1)
                ax[1].imshow(seg_pred[0], vmin=0, vmax=1)
            else:
                for i in range(self.n_classes):
                    ax[0][i].imshow(target[0, i].cpu().numpy(), vmin=0, vmax=1)
                    ax[1][i].imshow(seg_pred[0, i].cpu().numpy(), vmin=0, vmax=1)
            plt.show()

        return self.metrics
