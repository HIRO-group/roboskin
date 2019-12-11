import math
import time
import numpy as np
from mcp3208 import MCP3208

ADCONVERTER_BIT = 12
MIN_DISCRETE_SIGNAL = 0
MAX_DISCRETE_SIGNAL = math.pow(2, ADCONVERTER_BIT)
MAX_G = 3
VOLTAGE_SCALE  = 5/3.3

class Sensor(object):
    """
    Sensor Class
    """
    def __init__(self):
        pass

    def calibrate(self):
        """
        Calibrate the sensor
        """
        raise NotImplementedError()

    def read(self):
        """
        Fetch sensor data
        """
        raise NotImplementedError()

class ADXL335(Sensor):
    def __init__(self, xpin, ypin, zpin):
        super(ADXL335, self).__init__()
        self.adc = MCP3208()
        self.pins = [xpin, ypin, zpin]
        self.data = np.zeros(len(self.pins))
        self.calibrated = False
        self.mins = np.inf*np.ones(3)
        self.maxs = -np.inf*np.ones(3)
        self.bias = (MAX_DISCRETE_SIGNAL-MIN_DISCRETE_SIGNAL) / 2
        self.scale = (2*MAX_G) / (MAX_DISCRETE_SIGNAL-MIN_DISCRETE_SIGNAL)
        print('Make sure to supply 5V to the circuit & 3.3V to AD Converter Vref')

    def calibrate(self, n_loops=10000):
        """
        Depreciated
        """
        print('start calibrating')
        for i in range(n_loops):
            data = self.read()
            for j in range(3):
                self.mins[j] = min(self.mins[j], data[j])
                self.maxs[j] = max(self.maxs[j], data[j])
            print(self.mins, self.maxs)
        
        self.bias = (self.mins + self.maxs) / 2
        self.scale = 3 / (self.maxs-self.bias)
        self.calibrated = True

    def _read_raw(self):
        """
        returns python list of fetched sensor data
        """
        return [self.adc.read(pin) for pin in self.pins]

    def read(self):
        """
        returns numpy list of fetched sensor data
        """
        data = VOLTAGE_SCALE*np.array(self._read_raw())
        self.data = self.scale * (data - self.bias)
        return self.data

if __name__ == '__main__':
    accel_sensor = ADXL335(xpin=0, ypin=1, zpin=2)

    while True:
        data = accel_sensor.read()
        print(data[0], data[1], data[2])
        time.sleep(0.5)
