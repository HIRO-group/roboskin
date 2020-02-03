"""
read flexiforce sensor info example
"""
import time
# import numpy as np

from robotic_skin.sensor.flexiforce import FlexiForce

if __name__ == '__main__':
    # initialize accelerometer
    force_sensor = FlexiForce(pin=3)

    while True:
        data = force_sensor.read()
        print(data)
        time.sleep(0.5)
