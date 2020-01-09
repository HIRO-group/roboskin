import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time
import math
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
    line[0].set_data([], [])
    line[1].set_data([], [])

    return line,

def animate(i, X, Y1, Y2, vcnl4040, start):
    now = time.time()
    t = now - start
        
    X.append(t)
    Y1.append(vcnl4040.proximity)
    Y2.append(vcnl4040.lux)
    
    if len(X)>20:
        X = X[-20:]
        Y1 = Y1[-20:]
        Y2 = Y2[-20:]

    # assume x is larger than 0
    xlim = [0.9*np.min(X), 1.1*np.max(X)]
       
    ylim = [0, 500]

    line[0].set_data(X, Y1)
    line[1].set_data(X, Y2)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return line, 

if __name__ == '__main__':
    print('start animation')
    anim = animation.FuncAnimation(fig, animate, fargs=(X, Y1, Y2, vcnl4040, start), interval=int(1000/FREQ))
    plt.show()
    print('end animation')
