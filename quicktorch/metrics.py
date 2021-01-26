import math
import time
import torch
import torch.nn as nn
from sklearn.metrics import jaccard_score
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
    return all([item == 0 or item == 1 for item in y])


class MetricTracker():
    def __init__(self):
        self.metrics = OrderedDict()
        self.best_metrics = OrderedDict()
        self.epoch_count = 0
        self.batch_count = 0
        self.start_time = None
        self.batch_start = None
        self.stats = OrderedDict()

    def start(self):
        """Starts timer
        """
        self.start_time = time.time()
        self.batch_start = time.time()
        self.stats['avg_time'] = 0

    def reset(self):
        """Resets all metrics for new epoch
        """
        self.metrics.update({metric: torch.tensor(0.) for metric in self.metrics})
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

        This is an abstract method which MUST be overwritten.

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

    def is_best(self):
        """Checks if best metrics need to be updated.

        TODO add a way to update 'lower is better' metrics
        """
        self.epoch_count += 1
        # Rethink how best metrics are updated and compared.
        # Perhaps each metric can be a class
        # Define compare function?
        # Define master metric?
        if self.metrics[self.master_metric] > self.best_metrics[self.master_metric]:
            for metric in self.metrics:
                self.best_metrics[metric] = self.metrics[metric]
            self.best_metrics['epoch'] = self.epoch_count
            return True
        return False

    def finish(self):
        time_elapsed = time.time() - self.start_time
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best epoch: {}'.format(
            self._best_str()))

    def get_metrics(self):
        return self.typecheck_metrics(self.metrics)

    def get_best_metrics(self):
        return self.typecheck_metrics(self.best_metrics)

    @classmethod
    def typecheck_metrics(cls, m):
        out = OrderedDict()
        for key, val in m.items():
            if isinstance(val, torch.Tensor):
                out[key] = val.item()
            else:
                out[key] = val
        return out

    @classmethod
    def detect_metrics(cls, dataset):
        """Attempts to detect the most suitable metric tracker.
        """
        if type(dataset) is list or type(dataset) is tuple:
            dataset = dataset[0]
        data = dataset.dataset[0]
        if data[0].size(-1) == data[1].size(-1) and data[0].size(-2) == data[1].size(-2):
            if len(torch.unique(data[1])) > 2:
                return DenoisingTracker()
            return SegmentationTracker()

        # Get number of classes
        N = 0
        if hasattr(dataset, 'num_classes'):
            N = dataset.num_classes
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
        lbl_idx = target.max(dim=1)[1]
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
    def __init__(self):
        super().__init__()
        self.master_metric = "PSNR"
        self.metrics["PSNR"] = torch.tensor(0.)
        self.metrics["IoU"] = torch.tensor(0.)
        self.best_metrics = self.metrics.copy()
        self.mse_fn = nn.MSELoss()
        self.reset()

    def calculate(self, output, target):
        """Calculates metrics for given batch
        """
        mse = self.mse_fn(output, target)
        lbl = target.detach().cpu().flatten().numpy()
        pred = output.detach().cpu().round().flatten().numpy()
        pred = self._clip(pred, lbl)
        self.metrics['PSNR'] = (
            (self.batch_count * self.metrics['PSNR'] + 10 * math.log10(1 / mse.item())) /
            (self.batch_count + 1)
        )
        self.metrics['IoU'] = (
            (self.batch_count * self.metrics['IoU'] + jaccard_score(lbl, pred)) /
            (self.batch_count + 1)
        )
        return self.metrics

    @classmethod
    def _clip(cls, pred, lbl):
        return pred.clip(lbl.min(), lbl.max())


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
