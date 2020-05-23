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
        """
        return e, x
        """
        raise NotImplementedError()


class CombinedStopCondition(StopCondition):
    """
    Stop condition base class
    """
    def __init__(self, **kwargs):
        self.stop_conditions = []
        for k, v in kwargs.items():
            if not isinstance(v, StopCondition):
                raise ValueError('Only ErrorFunction class is allowed')
            setattr(self, k, v)
            self.stop_conditions.append(v)

    def initialize(self):
        pass

    def update(self, x, y, e):
        for stop_condition in self.stop_conditions:
            e, x = stop_condition.update(x, y, e)
        return e, x


class MaxCountStopCondition(StopCondition):
    """
    Stop condition base class
    """
    def __init__(self, count_limit=1000):
        self.count_limit = count_limit
        self.count = 0
        self.min_e = np.inf
        self.x = None

    def initialize(self):
        self.count = 0
        self.min_e = np.inf
        self.x = None

    def update(self, x, y, e):
        if self.x is None:
            self.x = x

        if e < self.min_e:
            self.min_e = e
            self.x = x

        self.count += 1
        if self.count >= self.count_limit:
            print(f'Reached Count Limit of {self.count_limit}')
            return 0.0, self.x

        return e, x


class PassThroughStopCondition(StopCondition):
    """
    PassThroughStopCondition class.
    """
    def __init__(self):
        super().__init__()

    def update(self, x, y, e):
        return e, x


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
                print(f'Reached DeltaX less than a threshold of {self.threshold}')
                return self.retval

        return e, x
