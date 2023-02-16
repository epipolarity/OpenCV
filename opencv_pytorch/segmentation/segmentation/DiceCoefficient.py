from segmentation.ConfusionMatrixBasedMetric import ConfusionMatrixBasedMetric
import numpy as np

# DiceCoefficient class inherited from ConfusionMatrixBasedMetric
class DiceCoefficient(ConfusionMatrixBasedMetric):
    """ Correct implementation of the Dice metric.

    Arguments:
        num_classes (int): number of evaluated classes.
        reduced_probs (bool): if True then argmax was applied to input predicts.
        normalized (bool): if normalized is True then confusion matrix will be normalized.
        ignore_indices (int or iterable): list of ignored classes indices.
    """

    # the core coefficient computation method
    def value(self):
        """ Return of the mean Dice and Dice per class.

        Returns:
            mdice (float32): mean dice.
            dice (list): list of dice coefficients per class.
        """
        # get confusion matrix value
        conf_matrix = self.conf_matrix.value()

        # check whether the list of indices to ignore is empty
        if self.ignore_indices is not None:
            # set column values of ignore classes to 0
            conf_matrix[:, self.ignore_indices] = 0
            # set row values of ignore classes to 0
            conf_matrix[self.ignore_indices, :] = 0

        # get TP, FP and FN values for Dice calculation using confusion matrix
        true_positive = np.diag(conf_matrix)
        false_positive = np.sum(conf_matrix, 0) - true_positive
        false_negative = np.sum(conf_matrix, 1) - true_positive

        # use errstate to handle the case of zero denominator value
        with np.errstate(divide='ignore', invalid='ignore'):
            # calculate dice by its formula
            dice = 2 * true_positive / (2 * true_positive + false_positive + false_negative)

        # check whether the list of indices to ignore is empty
        if self.ignore_indices is not None:
            # exclude ignore indices
            dice_valid_cls = np.delete(dice, self.ignore_indices)
            # get mean class dice coefficient ignoring NaN values
            mdice = np.nanmean(dice_valid_cls)
        else:
            # get mean class dice coefficient ignoring NaN values
            mdice = np.nanmean(dice)

        return mdice, dice