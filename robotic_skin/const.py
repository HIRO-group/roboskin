"""
This module serves to provide a variety of constants
for HIRO's robotic skin.

"""
# import nlopt

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
PATTERN_FREQ = [1, 1, 1, 1, 1, 1, 1]                # Hz
# PATTERN_FREQ = [1.1, 0.5, 0.65, 0.8, 0.8, 1.15, 1.15]                # Hz
# PATTERN_FREQ = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
# PATTERN_FREQ = [2, 2, 2, 2, 2, 2, 2]                # Hz

PATTERN_A = 0.4                 # rad/s
MAX_ANGULAR_VELOCITY = [1.5]*7      # rad/s
# MAX_ANGULAR_VELOCITY = [1.5, 0.8, 1, 1.3, 1.5, 2, 2]      # rad/s
# MAX_ANGULAR_VELOCITY = [1.5, 1.5, 1.5, 1.5, 1.75, 1.75, 1.75]      # rad/s
# MAX_ANGULAR_VELOCITY = 1.0, 0.5, 0.6, 0.8, 1.0, 1.2, 1.2
# MAX_ANGULAR_VELOCITY = [0.75, 0.25, 0.35, 0.55, 0.75, 0.9, 0.9]
T = 0.5                         # s
#GLOBAL_OPTIMIZER = nlopt.G_MLSL_LDS
#LOCAL_OPTIMIZER = nlopt.LN_NEWUOA
#GLOBAL_GRADIENT_OPTIMIZER = nlopt.GD_MLSL
#LOCAL_GRADIENT_OPTIMIZER = nlopt.LD_LBFGS
GRAVITATIONAL_CONSTANT = 9.81
