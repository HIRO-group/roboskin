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
ACCEL_FREQ = 1000               # Hz
GLOBAL_STOP_ROT = 0.035         # rad
GLOBAL_STOP_POS = 0.01          # m
LOCAL_STOP_ROT = 1e-10          # rad
LOCAL_STOP_POS = 1e-10          # m
PATTERN_FREQ = 2                # Hz
PATTERN_A = 0.4                 # rad/s
T = 0.5                         # s
GLOBAL_OPTIMIZER = nlopt.G_MLSL_LDS
LOCAL_OPTIMIZER = nlopt.LN_NELDERMEAD
