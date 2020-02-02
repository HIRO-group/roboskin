"""
This module serves to provide a variety of constants
for HIRO's robotic skin.

"""
import nlopt

ADXL335_XPIN = 0
ADXL335_YPIN = 1
ADXL335_ZPIN = 2
FLEXIFORCE_PIN = 3
N_POSE = 20
N_JOINT = 7
ACCEL_FREQ = 1000              # Hz
EXPLORE_SAMPLING_TIME = 1      # s
EXPLORE_SAMPLING_LENGTH = EXPLORE_SAMPLING_TIME * ACCEL_FREQ
EXPLORE_ANGULAR_VELOCITY = 0.5 # rad/s
ESTIMATE_SAMPLE_LENGTH = 1000  # 
GLOBAL_STOP = 0.1              # 
LOCAL_STOP = 1e-6              #
PATTERN_FREQ = 2               # Hz
PATTERN_A = 0.4                # rad/s
T = 0.5                        # s
GLOBAL_OPTIMIZER = nlopt.G_MLSL_LDS
LOCAL_OPTIMIZER = nlopt.LN_NELDERMEAD
