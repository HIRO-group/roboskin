"""
Read adxl335 sensor example
"""
import time
# import numpy as np

from robotic_skin.sensor.adxl335 import ADXL335

# initialize accelerometer
accel_sensor = ADXL335(xpin=0, ypin=1, zpin=2)

while True:
    data = accel_sensor.read()
    print(data[0], data[1], data[2])
    time.sleep(0.5)
