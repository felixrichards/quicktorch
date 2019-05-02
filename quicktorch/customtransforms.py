import torch


class ConvertType(object):
    def __init__(self, n_type):
        assert isinstance(n_type, torch.dtype)
        self.dtype = n_type

    def __call__(self, sample):
        return torch.tensor(sample, dtype=self.dtype)


class MakeCategorical(object):
    def __init__(self, n_classes=10):
        assert isinstance(n_classes, int)
        self.classes = n_classes

    def __call__(self, labels):
        if not isinstance(labels, torch.tensor):
            labels = torch.tensor(labels)
        
        if labels.dim() == 0:
            n_labels = torch.zeros(self.classes)
            n_labels[int(labels)] = 1
            return n_labels

        n_labels = torch.zeros(len(labels), self.classes)
        for i, label in enumerate(labels):
            n_labels[i, int(label)] = 1
        return n_labels

    def check_dim(self, t):
