import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
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
    line[0].set_data([], [])
    line[1].set_data([], [])
    line[2].set_data([], [])
        
    return line,

def animate(i, T, X, Y, Z, accelerometer, start, YMIN, YMAX):
    now = time.time()
    t = now - start
    data = accelerometer.read()
        
    # print('T:%03d, X:%04d, Y:%04d, Z:%04d'%(t, x, y, z))
        
    T.append(t)
    X.append(data[0])
    Y.append(data[1])
    Z.append(data[2])
    
    if len(T)>20:
        T = T[-20:]
        X = X[-20:]
        Y = Y[-20:]
        Z = Z[-20:]

    # assume x is larger than 0
    xlim = [np.min(T)-1, np.max(T)+1]
       
    # compute ylim
    ylim = [-2, 2]

    line[0].set_data(T, X)
    line[1].set_data(T, Y)
    line[2].set_data(T, Z)

    ax0.set_xlim(xlim)
    ax1.set_xlim(xlim)
    ax2.set_xlim(xlim)
    ax0.set_ylim(ylim)
    ax1.set_ylim(ylim)
    ax2.set_ylim(ylim)

    return line, 

print('start animation')
anim = animation.FuncAnimation(fig, animate, fargs=(T, X, Y, Z, accelerometer, start, YMIN, YMAX), interval=int(1000/FREQ))
plt.show()
print('end animation')
