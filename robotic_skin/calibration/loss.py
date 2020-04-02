import numpy as np


class Loss():
    """
    Loss function base class
    """
    def __init__(self):
        pass

    def __call__(self, x, axis=0):
        raise NotImplementedError()


class L1Loss(Loss):
    """
    L1 Loss
    """
    def __init__(self):
        super().__init__()

    def __call__(self, x, axis=0):
        return np.linalg.norm(x, axis=axis, ord=1)


class L2Loss(Loss):
    """
    L2Loss.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, x, axis=0):
        return np.linalg.norm(x, axis=axis)
