"""
plot flexiforce sensor info example
"""
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from robotic_skin.sensor.flexiforce import FlexiForce

force_sensor = FlexiForce(pin=3)

FREQ = 50

# initialize graph
fig = plt.figure()
ax = fig.add_subplot(111)
line, = ax.plot([], [])
X = []
Y = []

start = time.time()

def init():
    """
    animation function

    Returns
    ----------
    line
    """
    line.set_data([], [])

    return line

def animate(X_arr, Y_arr):
    """
    animation function

    Returns
    ----------
    line
    """
    now = time.time()
    t = now - start
    data = force_sensor.read()
        
    X_arr.append(t)
    Y_arr.append(data)
    
    if len(X_arr) > 20:
        X_arr = X_arr[-20:]
        Y_arr = Y_arr[-20:]

    # assume x is larger than 0
    xlim = [np.min(X_arr)-1, np.max(X_arr)+1]
       
    ylim = [0, 4000]

    line.set_data(X_arr, Y_arr)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return line

if __name__ == '__main__':
    print('start animation')
    anim = animation.FuncAnimation(fig, animate, fargs=(X, Y, force_sensor, start), interval=int(1000/FREQ))
    plt.show()
    print('end animation')
