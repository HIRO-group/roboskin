import numpy as np


class Loss():
    def __init__(self):
        pass

    def __call__(self, x, axis=0):
        raise NotImplementedError()


class L1Loss(Loss):
    def __init__(self):
        super().__init__()

    def __call__(self, x, axis=0):
        return np.linalg.norm(x, axis=axis, ord=1)


class L2Loss(Loss):
    def __init__(self):
        super().__init__()

    def __call__(self, x, axis=0):
        return np.linalg.norm(x, axis=axis)
