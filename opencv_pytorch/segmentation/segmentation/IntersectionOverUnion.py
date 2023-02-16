from trainer.base_metric import BaseMetric
from segmentation.ConfusionMatrix import ConfusionMatrix
import numpy as np

# create intersection over union class
class IntersectionOverUnion(BaseMetric):
    """
        Implementation of the Intersection over Union metric.

        Arguments:
            num_classes (int): number of evaluated classes.
            reduced_probs (bool): if True, then argmax was applied to the input predictions.
            normalized (bool): if normalized is True, then confusion matrix will be normalized.
            ignore_indices (int or iterable): list of ignored classes indices.
    """
    def __init__(self, num_classes, reduced_probs=False, normalized=False, ignore_indices=None):
        # created a normalized confusion matrix with num_classes
        self.conf_matrix = ConfusionMatrix(num_classes=num_classes, normalized=normalized)
        self.reduced_probs = reduced_probs

        # check whether ignored classes exist
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
        """
            Reset the Confusion Matrix
        """
        self.conf_matrix.reset()

    def update_value(self, pred, target):
        """ Add sample to the Confusion Matrix.

            Arguments:
                pred (torch.Tensor() or numpy.ndarray): predicted mask.
                target (torch.Tensor() or numpy.ndarray): ground-truth mask.
        """
        if not self.reduced_probs:
            pred = pred.argmax(dim=1)
        self.conf_matrix.add(pred, target)

    def get_metric_value(self):
        """
            Return mIOU and IOU per class.

            Returns:
                miou (float32): mean intersection over union.
                iou (list): list of intersection over union per class.
        """
        # get confusion matrix value
        conf_matrix = self.conf_matrix.value()

        # check whether the list of indices to ignore is empty
        if self.ignore_indices is not None:
            # set column values of ignore classes to 0
            conf_matrix[:, self.ignore_indices] = 0
            # set row values of ignore classes to 0
            conf_matrix[self.ignore_indices, :] = 0

        # get TP, FP and FN values
        true_positive = np.diag(conf_matrix)
        false_positive = np.sum(conf_matrix, 0) - true_positive
        false_negative = np.sum(conf_matrix, 1) - true_positive

        # use errstate to handle the case of zero denominator value
        with np.errstate(divide='ignore', invalid='ignore'):
            # calculate iou by its formula
            iou = true_positive / (true_positive + false_positive + false_negative)

        # check whether the list of indices to ignore is empty
        if self.ignore_indices is not None:
            # exclude ignore indices
            iou_valid_cls = np.delete(iou, self.ignore_indices)
            # get mean class iou value ignoring NaN values
            miou = np.nanmean(iou_valid_cls)
        else:
            # get mean class iou value ignoring NaN values
            miou = np.nanmean(iou)
        return {"mean_iou": miou, "iou": iou}