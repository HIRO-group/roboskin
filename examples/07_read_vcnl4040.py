"""
Read vcnl4040 sensor example
"""

# pylint: skip-file   
 
import time
import board
import busio
import adafruit_vcnl4040

i2c = busio.I2C(board.SCL, board.SDA)
sensor = adafruit_vcnl4040.VCNL4040(i2c)

while True:
    print('Proximity: ', sensor.proximity)
    print('Light: ', sensor.lux)
    time.sleep(0.5)
