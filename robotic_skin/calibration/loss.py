import numpy as np


class Loss():
    """
    Loss function base class
    """
    def __init__(self, hyper_params):
        self.hyper_params = hyper_params

    def __call__(self, x):
        raise NotImplementedError()


class L1Loss(Loss):
    """
    L1 Loss
    """
    def __init__(self, hyper_params):
        super().__init__(hyper_params)

    def __call__(self, x):
        return np.mean(np.linalg.norm(x, axis=1))


class L2Loss(Loss):
    """
    L2Loss.
    """
    def __init__(self, hyper_params):
        super().__init__(hyper_params)

    def __call__(self, x):
        raise NotImplementedError()
