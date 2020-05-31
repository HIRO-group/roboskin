import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


class Loss():
    """
    Loss function base class
    """

    def __init__(self):
        pass

    def __call__(self, x_estimated, x_target, axis=0):
        raise NotImplementedError()


class L1Loss(Loss):
    """
    L1 Loss
    """

    def __init__(self):
        super().__init__()

    def __call__(self, x_estimated, x_target, axis=0):
        x = np.abs(x_estimated - x_target)
        return np.mean(np.sum(x))


class L2Loss(Loss):
    """
    L2Loss.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, x_estimated, x_target, axis=0):
        x = (x_estimated - x_target) ** 2
        return np.mean(np.sum(x))


class MeanSquareLoss(Loss):
    """
    mean square loss
    """
    def __init__(self):
        super().__init__()

    def __call__(self, x_estimated, x_target, axis=0):
        return mean_squared_error(x_target, x_estimated)


class MeanAbsoluteLoss(Loss):
    """
    mean absolute loss
    """
    def __init__(self):
        super().__init__()

    def __call__(self, x_estimated, x_target, axis=0):
        return mean_absolute_error(x_target, x_estimated)


class L1LossTorch(Loss):
    """
    pytorch's l1loss for tensors.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, x_estimated, x_target):
        loss = nn.L1Loss()
        output = loss(x_estimated, x_target)
        return output


class L2LossTorch(Loss):
    """
    pytorch's l2loss for tensors.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, x_estimated, x_target):
        loss = nn.MSELoss()
        output = loss(x_estimated, x_target)
        return output


class SmoothL1LossTorch(Loss):
    """
    pytorch's *smooth* l1loss for tensors.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, x_estimated, x_target):
        loss = nn.SmoothL1Loss()
        output = loss(x_estimated, x_target)
        return output
