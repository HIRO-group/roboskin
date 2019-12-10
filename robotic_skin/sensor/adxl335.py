import numpy as np
from mcp3208 import MCP3208

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

    def _read(self):
        """
        returns python list of fetched sensor data
        """
        return [self.adc.read(pin) for pin in self.pins]

    def read(self):
        """
        returns numpy list of fetched sensor data
        """
        self.data = np.array(self._read())
        return self.data
