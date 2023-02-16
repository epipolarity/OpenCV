import torch
import numpy as np


class ConfusionMatrix:
    """ Implementation of Confusion Matrix.g

    Arguments:
        num_classes (int): number of evaluated classes.
        normalized (bool): if normalized is True then confusion matrix will be normalized.
    """
    def __init__(self, num_classes, normalized=False):
        self.num_classes = num_classes
        self.normalized = normalized
        self.conf = np.ndarray((num_classes, num_classes), np.int32)
        self.reset()

    def reset(self):
        """ Reset of the Confusion Matrix.
        """
        self.conf.fill(0)

    def add(self, pred, target):
        """ Add sample to the Confusion Matrix.

        Arguments:
            pred (torch.Tensor() or numpy.ndarray): predicted mask.
            target (torch.Tensor() or numpy.ndarray): ground-truth mask.
        """
        if torch.is_tensor(pred):
            pred = pred.detach().cpu().numpy()
        if torch.is_tensor(target):
            target = target.detach().cpu().numpy()

        valid_indices = np.where((target >= 0) & (target < self.num_classes))
        pred = pred[valid_indices]
        target = target[valid_indices]

        replace_indices = np.vstack((target.flatten(), pred.flatten())).T

        conf, _ = np.histogramdd(
            replace_indices,
            bins=(self.num_classes, self.num_classes),
            range=[(0, self.num_classes), (0, self.num_classes)]
        )

        self.conf += conf.astype(np.int32)

    def value(self):
        """ Return of the Confusion Matrix.

        Returns:
            numpy.ndarray(num_classes, num_classes): confusion matrix.
        """
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        return self.conf