import torch
"""
This module contains multiple PIL image transformations
compatible with torchvision.transforms.Compose.
"""


class ConvertType(object):
    """Converts the dtype of a given tensor.

    Args:
        n_type (torch.dtype): Desired dtype to convert to.
    """
    def __init__(self, n_type):
        assert isinstance(n_type, torch.dtype)
        self.dtype = n_type

    def __call__(self, sample):
        """
        Args: sample (torch.tensor): Input to be converted.

        Returns:
            torch.tensor: Input with converted dtype.
        """
        return torch.tensor(sample, dtype=self.dtype)


class MakeCategorical(object):
    """Converts a label tensor to categorical/onehot style

    E.g. `MakeCategorical()([2,5]) = [[0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1]]`

    Args:
        n_classes (int, optional): Number of classes. Defaults to 10.
    """
    def __init__(self, n_classes=10):
        assert isinstance(n_classes, int)
        self.classes = n_classes

    def __call__(self, labels):
        """
        Args:
            labels (torch.tensor): Input to be made categorical.

        Returns:
            torch.tensor: A categorical representation of the input labels.
        """
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)

        if labels.dim() == 0:
            n_labels = torch.zeros(self.classes)
            n_labels[int(labels)] = 1
            return n_labels

        n_labels = torch.zeros(len(labels), self.classes)
        for i, label in enumerate(labels):
            n_labels[i, int(label)] = 1
        return n_labels
