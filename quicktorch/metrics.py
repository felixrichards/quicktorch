import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from sklearn.metrics import jaccard_score, f1_score
from skimage.metrics import adapted_rand_error
from collections import OrderedDict


def dict2str(dictionary):
    sorted_keys = sorted([key for key in dictionary.keys()])
    return ' '.join(
        ['{}: {}.'.format(key.title(), _format_val(key, dictionary[key]))
         for key in sorted_keys]
    )


def _format_val(key, val):
    if key == "epoch" or type(val) is int:
        return str(int(val))
    if "time" in key:
        return '{:.4f}'.format(val)
    return '{:.4f}'.format(val)


def _is_one_hot(y):
    #check  item is array-like
    return all([item == 0 or item == 1 for item in y])


class MetricTracker():
    """Base metric tracker class.

    TODO separate metric dicts for phases
    """
    def __init__(self, Writer=None):
        self.metrics = OrderedDict()
        self.best_metrics = OrderedDict()
        self.epoch_count = 0
        self.batch_count = 0
        self.start_time = None
        self.batch_start = None
        self.stats = OrderedDict()
        self.Writer = Writer

    def start(self, phases=None):
        """Starts timer
        """
        self.start_time = time.time()
        self.batch_start = time.time()
        self.stats['avg_time'] = 0
        if self.Writer is not None:
            self.Writer.start({**self.get_metrics(), **self.get_stats()}, phases=phases)

    def reset(self, phase=None, loss=None):
        """Resets all metrics for new epoch
        """
        if self.Writer is not None:
            if loss is not None:
                self.metrics['loss'] = loss
            self.Writer.add({**self.get_metrics(), **self.get_stats()}, phase=phase)
        self.clear_metrics()

    def clear_metrics(self):
        """Clear metrics
        """
        self.reset_dict(self.metrics)
        self.batch_start = time.time()
        self.batch_count = 0
        self.reset_buffers()

    def reset_buffers(self):
        """Resets any variables used for metric calculations for new epoch

        This is an abstract method which can be overwritten if needed.
        """
        pass

    def update(self, output, target):
        """Updates metrics
        """
        with torch.no_grad():
            self.calculate(output, target)
        if self.start_time is not None:
            self.stats['avg_time'] = (
                (self.stats['avg_time'] * self.batch_count + time.time() - self.batch_start) /
                (self.batch_count + 1)
            )
            self.batch_start = time.time()
        self.batch_count += 1

    def calculate(self, output, target):
        """Calculates metrics for a given batch

        This is an abstract method which MUST be overridden.

        TODO implement individual classes for metrics that makes this unnecessary.
        """
        raise NotImplementedError("Must implement calculate method.")

    def progress_str(self):
        """Returns a string for current metrics
        """
        return dict2str(self.metrics)

    def stats_str(self):
        """Returns a string for current metrics
        """
        return dict2str(self.stats)

    def _best_str(self):
        """Returns a string for best metrics
        """
        return dict2str(self.best_metrics)

    def show_best(self):
        """Prints best metrics
        """
        print('Best epoch so far: {}'
              .format(self._best_str()))

    def is_best(self, loss=None):
        """Checks if best metrics need to be updated.

        TODO add a way to update 'lower is better' metrics
        """
        self.epoch_count += 1
        # Rethink how best metrics are updated and compared.
        # Perhaps each metric can be a class
        # Define compare function?
        # Define master metric?
        if self.metrics[self.master_metric] >= self.best_metrics[self.master_metric]:
            for metric in self.metrics:
                self.best_metrics[metric] = self.metrics[metric]
            self.best_metrics['epoch'] = self.epoch_count
            return True
        return False

    def finish(self):
        time_elapsed = time.time() - self.start_time
        time_elapsed = 0
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best epoch: {}'.format(
            self._best_str()))

    def get_stats(self):
        return self.typecheck_metrics(self.stats)

    def get_metrics(self):
        return self.typecheck_metrics(self.metrics)

    def get_best_metrics(self):
        return self.typecheck_metrics(self.best_metrics)

    def batch_average(self, metric, metric_key):
        return (
            (self.batch_count * self.metrics[metric_key] + metric) /
            (self.batch_count + 1)
        )

    @classmethod
    def reset_dict(cls, d):
        d.update({di: torch.tensor(0.) for di in d})
        return d

    @classmethod
    def typecheck_metrics(cls, m):
        out = OrderedDict()
        for key, val in m.items():
            if isinstance(val, torch.Tensor):
                out[key] = val.item()
            else:
                out[key] = val
            if math.isnan(val):
                out[key] = 0
        return out

    @classmethod
    def detect_metrics(cls, dataloader):
        """Attempts to detect the most suitable metric tracker.
        """
        if type(dataloader) is list or type(dataloader) is tuple:
            dataloader = dataloader[0]
        data = dataloader.dataset[0]

        if data[1].ndim > 0:
            if data[0].size(-1) == data[1].size(-1) and data[0].size(-2) == data[1].size(-2):
                if len(torch.unique(data[1])) > 2:
                    return DenoisingTracker()
                return SegmentationTracker()

        # Get number of classes
        N = 0
        print(dataloader.dataset.num_classes)
        if hasattr(dataloader.dataset, 'num_classes'):
            N = dataloader.dataset.num_classes
            return ClassificationTracker(N)
        if _is_one_hot(data[1]):
            N = len(data[1])
            return ClassificationTracker(N)

        return RegressionTracker()


class ClassificationTracker(MetricTracker):
    """Tracks metrics for classification performance.

    Args:
        n_classes (int): Number of distinct classes.
    """
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.master_metric = "accuracy"
        self.metrics["accuracy"] = torch.tensor(0.)
        self.metrics["precision"] = torch.tensor(0.)
        self.metrics["recall"] = torch.tensor(0.)
        self.metrics["loss"] = torch.tensor(0.)
        self.best_metrics = self.metrics.copy()
        self.reset()

    def reset_buffers(self):
        """Resets any variables used for metric calculations for new epoch.
        """
        self.confusion = torch.zeros(self.n_classes, self.n_classes)

    def calculate(self, output, target):
        """Calculates metrics for given batch
        """
        out_idx = output.max(dim=1)[1]
        if target.ndim > 1:
            lbl_idx = target.max(dim=1)[1]
        else:
            lbl_idx = target
        if self.confusion is not None:
            for j, k in zip(out_idx, lbl_idx):
                self.confusion[j, k] += 1
        corr = self.confusion.diag()
        self.metrics["accuracy"] = corr.sum() / self.confusion.sum()
        self.metrics["precision"] = (corr / self.confusion.sum(1)).mean()
        self.metrics["recall"] = (corr / self.confusion.sum(0)).mean()
        return self.metrics


class SegmentationTracker(MetricTracker):
    """Tracks metrics for segmentation performance.
    """
    def __init__(self, full_metrics=False):
        super().__init__()
        self.master_metric = "IoU"
        self.metrics["PSNR"] = torch.tensor(0.)
        self.metrics["IoU"] = torch.tensor(0.)
        self.metrics["Dice"] = torch.tensor(0.)
        self.full_metrics = full_metrics
        if full_metrics:
            self.metrics["error"] = torch.tensor(0.)
            self.metrics["precision"] = torch.tensor(0.)
            self.metrics["recall"] = torch.tensor(0.)
        self.best_metrics = self.metrics.copy()
        self.mse_fn = nn.MSELoss()
        self.reset()

    def calculate(self, output, target):
        """Calculates metrics for given batch
        """
        lbl = target.detach().cpu().numpy()
        pred = output.detach()
        pred = torch.sigmoid(pred)
        mse = self.mse_fn(pred, target)
        pred = pred.cpu().round().numpy()
        if self.full_metrics:
            error, precision, recall = adapted_rand_error(lbl.astype('int'), pred.astype('int'))
            self.metrics['error'] = self.batch_average(error, 'error')
            self.metrics['precision'] = self.batch_average(precision, 'precision')
            self.metrics['recall'] = self.batch_average(recall, 'recall')
        self.metrics['PSNR'] = self.batch_average(10 * math.log10(1 / mse.item()), 'PSNR')
        self.metrics['IoU'] = self.batch_average(iou(pred, lbl), 'IoU')
        self.metrics["Dice"] = self.batch_average(dice(pred, lbl), 'IoU')
        return self.metrics


class MultiClassSegmentationTracker(MetricTracker):
    """Tracks metrics for segmentation performance.
    """
    def __init__(self, full_metrics=False, n_classes=10):
        super().__init__()
        self.master_metric = "IoU"
        self.metrics["PSNR"] = torch.tensor(0.)
        self.metrics["IoU"] = torch.tensor(0.)
        self.metrics["Dice"] = torch.tensor(0.)
        self.full_metrics = full_metrics
        self.best_metrics = self.metrics.copy()
        self.iou_fn = torchmetrics.IoU(n_classes)
        self.dice_fn = torchmetrics.F1(n_classes, mdmc_average='samplewise')
        self.reset()

    def calculate(self, output, target):
        """Calculates metrics for given batch
        """
        target = target.detach()
        output = output.detach()
        output = F.softmax(output, dim=1)
        output = torch.argmax(output, dim=1)

        self.metrics['IoU'] = self.batch_average(self.iou_fn(output, target), 'IoU')
        self.metrics["Dice"] = self.batch_average(self.dice_fn(output, target), 'Dice')
        return self.metrics


class DenoisingTracker(MetricTracker):
    """Tracks metrics for denoising performance.
    """
    def __init__(self):
        super().__init__()
        self.master_metric = "PSNR"
        self.metrics["PSNR"] = torch.tensor(0.)
        self.best_metrics = self.metrics.copy()
        self.mse_fn = nn.MSELoss()
        self.reset()

    def calculate(self, output, target):
        """Calculates metrics for given batch
        """
        mse = self.mse_fn(output, target)
        self.metrics['PSNR'] = (
            (self.batch_count * self.metrics['PSNR'] + 10 * math.log10(1 / mse.item())) /
            (self.batch_count + 1)
        )
        return self.metrics


class RegressionTracker(MetricTracker):
    """Tracks metrics for regression performance.
    """
    def __init__(self):
        super().__init__()
        self.master_metric = "RMSE"
        self.metrics["RMSE"] = torch.tensor(0.)
        self.best_metrics = self.metrics.copy()
        self.mse_fn = nn.MSELoss()
        self.reset()

    def calculate(self, output, target):
        """Calculates metrics for given batch
        """
        rmse = torch.sqrt(self.mse_fn(output, target))
        self.metrics['RMSE'] = (
            (self.batch_count * self.metrics['RMSE'] + rmse) /
            (self.batch_count + 1)
        )
        return self.metrics


def _clip(pred):
    return pred.clip(0, 1)


def iou(pred, lbl, to_mask=None):
    pred = pred.flatten()
    lbl = lbl.flatten()
    if to_mask is not None:
        pred = to_mask(pred, lbl)
    return jaccard_score(lbl, pred, zero_division=0)


def dice(pred, lbl, to_mask=None):
    pred = pred.flatten()
    lbl = lbl.flatten()
    if to_mask is not None:
        pred = to_mask(pred)
    return f1_score(lbl, pred, zero_division=0)
