from  segmentation.ConfusionMatrix import ConfusionMatrix

class ConfusionMatrixBasedMetric:
    """ Implementation of base class for Confusion Matrix based metrics.

    Arguments:
        num_classes (int): number of evaluated classes.
        reduced_probs (bool): if True then argmax was applied to input predicts.
        normalized (bool): if normalized is True then confusion matrix will be normalized.
        ignore_indices (int or iterable): list of ignored classes index.
    """
    def __init__(self, num_classes, reduced_probs=False, normalized=False, ignore_indices=None):
        self.conf_matrix = ConfusionMatrix(num_classes=num_classes, normalized=normalized)
        self.reduced_probs = reduced_probs

        if ignore_indices is None:
            self.ignore_indices = None
        elif isinstance(ignore_indices, int):
            self.ignore_indices = (ignore_indices, )
        else:
            try:
                self.ignore_indices = tuple(ignore_indices)
            except TypeError:
                raise ValueError("'ignore_indices' must be an int or iterable")

    def reset(self):
        """ Reset of the Confusion Matrix
        """
        self.conf_matrix.reset()

    def add(self, pred, target):
        """ Add sample to the Confusion Matrix.

        Arguments:
            pred (torch.Tensor() or numpy.ndarray): predicted mask.
            target (torch.Tensor() or numpy.ndarray): ground-truth mask.
        """
        if not self.reduced_probs:
            pred = pred.argmax(dim=1)
        self.conf_matrix.add(pred, target)