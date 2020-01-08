import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
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
    line.set_data([], [])

    return line,

def animate(i, X, Y, force_sensor, start):
    now = time.time()
    t = now - start
    data = force_sensor.read()
        
    X.append(t)
    Y.append(data)
    
    if len(X)>20:
        X = X[-20:]
        Y = Y[-20:]

    # assume x is larger than 0
    xlim = [0.9*np.min(X), 1.1*np.max(X)]
       
    ylim = [0, 4000]

    line.set_data(X, Y)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return line, 

# if __name__ == "__main__":
print('start animation')
anim = animation.FuncAnimation(fig, animate, fargs=(X, Y, force_sensor, start), interval=int(1000/FREQ))
plt.show()
print('end animation')
