"""
Plot results from vcnl4040 sensor.
"""

# pylint: skip-file

import time
# import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import board
import busio
import adafruit_vcnl4040

i2c = busio.I2C(board.SCL, board.SDA)
vcnl4040 = adafruit_vcnl4040.VCNL4040(i2c)

FREQ = 50

# initialize graph
fig = plt.figure()
ax = fig.add_subplot(111)
l1, = ax.plot([], [])
l2, = ax.plot([], [])
line = [l1, l2]
X = []
Y1 = []
Y2 = []

start = time.time()

def init():
    """
    set empty data for line

    Returns
    ----------
    line
    """
    line[0].set_data([], [])
    line[1].set_data([], [])

    return line

def animate(X_arr, Y1_arr, Y2_arr):
    """
    animation function

    Returns
    ----------
    line
    """
    now = time.time()
    t = now - start
        
    X_arr.append(t)
    Y1_arr.append(vcnl4040.proximity)
    Y2_arr.append(vcnl4040.lux)
    
    if len(X_arr) > 20:
        X_arr = X_arr[-20:]
        Y1_arr = Y1_arr[-20:]
        Y2_arr = Y2_arr[-20:]

    # assume x is larger than 0
    xlim = [np.min(X_arr)-1, np.max(X_arr)+1]
       
    ylim = [0, 500]

    line[0].set_data(X_arr, Y1_arr)
    line[1].set_data(X_arr, Y2_arr)
    ax.set_xlim(xlim)
    # set axis limits
    ax.set_ylim(ylim)

    return line 

if __name__ == '__main__':
    print('start animation')
    anim = animation.FuncAnimation(fig, animate, fargs=(X, Y1, Y2, vcnl4040, start), interval=int(1000/FREQ))
    plt.show()
    print('end animation')
