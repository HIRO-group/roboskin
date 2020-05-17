import numpy as np


class StopCondition():
    """
    Stop condition base class
    """
    def __init__(self):
        pass

    def initialize(self):
        pass

    def update(self, x, y, e):
        raise NotImplementedError()


class PassThroughStopCondition(StopCondition):
    """
    PassThroughStopCondition class.
    """
    def __init__(self):
        super().__init__()

    def update(self, x, y, e):
        return e


class DeltaXStopCondition(StopCondition):
    """
    DeltaXStopCondition class. Keeps track on the
    differences in x from iteration to iteration,
    until the updates are very small.
    """
    def __init__(self, windowsize=10, threshold=0.001, retval=0.00001):
        super().__init__()

        self.initialize()
        self.windowsize = windowsize
        self.threshold = threshold
        self.retval = retval

    def initialize(self):
        self.prev_x = None
        self.xdiff = None
        self.xdiffs = np.array([])

    def update(self, x, y, e):
        if self.prev_x is None:
            self.xdiff = None
            self.prev_x = np.array(x)
        else:
            self.xdiff = np.linalg.norm(np.array(x) - self.prev_x)
            self.prev_x = np.array(x)

        if self.xdiff is not None:
            self.xdiffs = np.append(self.xdiffs, self.xdiff)
        if len(self.xdiffs) >= self.windowsize:
            if np.mean(self.xdiffs[-(self.windowsize+1):-1]) <= self.threshold:
                return self.retval

        return e
