import numpy as np
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
        x = x_estimated - x_target
        return np.mean(np.linalg.norm(x, axis=axis, ord=1))


class L2Loss(Loss):
    """
    L2Loss.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, x_estimated, x_target, axis=0):
        x = x_estimated - x_target
        return np.mean(np.linalg.norm(x, axis=axis))


class MeanSquareLoss(Loss):
    def __init__(self):
        super().__init__()

    def __call__(self, x_estimated, x_target, axis=0):
        return mean_squared_error(x_target, x_estimated)


class MeanAbsoluteLoss(Loss):
    def __init__(self):
        super().__init__()

    def __call__(self, x_estimated, x_target, axis=0):
        return mean_absolute_error(x_target, x_estimated)
