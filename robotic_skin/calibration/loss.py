class Loss():
    def __init__(self, hyper_params):
        self.hyper_params = hyper_params

    def __call__(self, x):
        raise NotImplementedError()

class L1Loss(Loss):
    def __init__(self, hyper_params):
        super().__init__(hyper_params)

    def __call__(self, x):
        raise NotImplementedError()

class L2Loss(Loss):
    def __init__(self, hyper_params):
        super().__init__(hyper_params)

    def __call__(self, x):
        raise NotImplementedError()
