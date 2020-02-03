"""
Plot adxl335 sensor info example
"""
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from robotic_skin.sensor.adxl335 import ADXL335

accelerometer = ADXL335(xpin=0, ypin=1, zpin=2)

FREQ = 50

# initialize graph
fig = plt.figure()
ax0 = fig.add_subplot(311)
ax1 = fig.add_subplot(312)
ax2 = fig.add_subplot(313)
l0, = ax0.plot([], [])
l1, = ax1.plot([], [])
l2, = ax2.plot([], [])
line = [l0, l1, l2]
X = []
Y = []
Z = []
T = []

start = time.time()
YMAX = -np.inf
YMIN = np.inf

def init():
    """
    set empty data for line

    Returns
    ----------
    line
    """
    line[0].set_data([], [])
    line[1].set_data([], [])
    line[2].set_data([], [])
        
    return line

def animate(T_arr, X_arr, Y_arr, Z_arr):
    """
    animation function

    Returns
    ----------
    line
    """
    now = time.time()
    t = now - start
    data = accelerometer.read()
        
    # print('T:%03d, X:%04d, Y:%04d, Z:%04d'%(t, x, y, z))
        
    T_arr.append(t)
    X_arr.append(data[0])
    Y_arr.append(data[1])
    Z_arr.append(data[2])
    
    if len(T_arr) > 20:
        T_arr = T_arr[-20:]
        X_arr = X_arr[-20:]
        Y_arr = Y_arr[-20:]
        Z_arr = Z_arr[-20:]

    # assume x is larger than 0
    xlim = [np.min(T_arr)-1, np.max(T_arr)+1]
       
    # compute ylim
    ylim = [-2, 2]

    line[0].set_data(T_arr, X_arr)
    line[1].set_data(T_arr, Y_arr)
    line[2].set_data(T_arr, Z_arr)

    ax0.set_xlim(xlim)
    ax1.set_xlim(xlim)
    ax2.set_xlim(xlim)
    ax0.set_ylim(ylim)
    ax1.set_ylim(ylim)
    ax2.set_ylim(ylim)

    return line

print('start animation')
anim = animation.FuncAnimation(fig, animate, fargs=(T, X, Y, Z, accelerometer, start, YMIN, YMAX), interval=int(1000/FREQ))
plt.show()
print('end animation')
