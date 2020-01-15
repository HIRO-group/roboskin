import math
import time
import numpy as np
from robotic_skin.sensor import Sensor
from mcp3208 import MCP3208


class ADXL335(Sensor):
    def __init__(self, xpin, ypin, zpin):
        super(ADXL335, self).__init__()
        # We are using MPC3208 library from Pypi, thank God someone wrote it
        self.adc = MCP3208()
        self.pins = [xpin, ypin, zpin]
        # self.data is holder for all our accelerations in x,y,z directions
        self.data = np.zeros(len(self.pins))
        self.calibrate = False
        print('Make sure to supply 5V to the circuit & 3.3V to AD Converter Vref')

    def calibrate(self) -> None:
        # How dumb can it be to write some shit like this that calibrate = True in calibrate function?
        # Well we gotta calibrate it before. It isn't automated, which I hate, but I gotta do what I gotta do
        # This is basically implemented to satisfy the Sensor class, as I have to override the class for sure
        # TODO: Discuss with @kandai about calibrate. Wanna keep it a property or what?
        self.calibrate = True

    def _read_raw(self):
        """
        returns python list of fetched sensor data. The data ranges from [0, 4096]
        It's basically an ADC output. As MPC3208 is a 12 bit ADC, 2^12 = 4096, hence the 4096 is the
        upper limit
        """
        return [self.adc.read(pin) for pin in self.pins]

    def read(self):
        # Read raw ADC values from the sensor
        data = np.array(self._read_raw())
        # How this equation came up, look for my blog:
        # https://krishnachaitanya9.github.io/posts/RPi_Calibrating_ADXL335_Accelerometer/
        # There is a more detailed explanation
        # which can't be fir into comments. So please go look, and lemme know if there is any problem with the logic
        self.data = ((((3.3 / 4096) * data) - 2) * 3.1) + 1
        # Returning back the data
        return self.data


if __name__ == '__main__':
    accel_sensor = ADXL335(xpin=0, ypin=1, zpin=2)
    # TODO: Do we need this? Can't this be automatically done when sensor is initialized?
    accel_sensor.calibrate
    while True:
        data = accel_sensor.read()
        print("Z Acceleration: ", data[2])
        time.sleep(0.5)
