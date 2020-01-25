import sys
import nlopt

class MyConstBaseClass(object):
    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise NameError("Can't rebind const(%s)"%name)
        self.__dict__[name] = value

class _const(MyConstBaseClass):
    def __init__(self):
        # Add constants here
        self.ADXL335_XPIN = 0
        self.ADXL335_YPIN = 1
        self.ADXL335_ZPIN = 2
        self.FLEXIFORCE_PIN = 3
        self.N_POSE = 20
        self.N_JOINT = 7
        self.ACCEL_FREQ = 1000              # Hz
        self.EXPLORE_SAMPLING_TIME = 1      # s
        self.EXPLORE_SAMPLING_LENGTH = self.EXPLORE_SAMPLING_TIME * self.ACCEL_FREQ
        self.EXPLORE_ANGULAR_VELOCITY = 0.5 # rad/s
        self.ESTIMATE_SAMPLE_LENGTH = 1000  # 
        self.GLOBAL_STOP = 0.1              # 
        self.LOCAL_STOP = 1e-6              #
        self.PATTERN_FREQ = 2               # Hz
        self.PATTERN_A = 0.4                # rad/s
        self.T = 0.5                        # s
        self.GLOBAL_OPTIMIZER = nlopt.G_MLSL_LDS
        self.LOCAL_OPTIMIZER = nlopt.LN_NELDERMEAD

sys.modules[__name__] = _const()