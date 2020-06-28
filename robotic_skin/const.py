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
GLOBAL_STOP = 0.0001
ROT_GLOBAL_STOP = 0.01         #
POS_GLOBAL_STOP = 0.01          #
LOCAL_STOP = 1e-10              #
GLOBAL_XTOL = 1e-2
PATTERN_FREQ = 1                # Hz
PATTERN_A = 0.4                 # rad/s
MAX_ANGULAR_VELOCITY = 1.5      # rad/s
T = 0.5                         # s
GLOBAL_OPTIMIZER = nlopt.G_MLSL_LDS
LOCAL_OPTIMIZER = nlopt.LN_NEWUOA
GLOBAL_GRADIENT_OPTIMIZER = nlopt.GD_MLSL
LOCAL_GRADIENT_OPTIMIZER = nlopt.LD_LBFGS
GRAVITATIONAL_CONSTANT = 9.80
