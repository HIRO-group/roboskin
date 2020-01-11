import math
import time
import numpy as np
from mcp3208 import MCP3208

# TODO: Calibration should either be done or should be read from text file. Make changes in code to enforce that
'''
Below constants will be deprecated in near future
'''
ADCONVERTER_BIT = 12
MIN_DISCRETE_SIGNAL = 0
MAX_DISCRETE_SIGNAL = math.pow(2, ADCONVERTER_BIT)
MAX_G = 3
VOLTAGE_SCALE = 5 / 3.3


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
        self.mins = np.inf * np.ones(3)
        self.maxs = -np.inf * np.ones(3)
        self.bias = (MAX_DISCRETE_SIGNAL - MIN_DISCRETE_SIGNAL) / 2
        self.scale = (2 * MAX_G) / (MAX_DISCRETE_SIGNAL - MIN_DISCRETE_SIGNAL)
        self.x_min = -3
        self.x_max = 3
        self.y_min = -3
        self.y_max = 3
        self.z_min = -3
        self.z_max = 3
        self.out_min = -3
        self.out_max = 3
        print('Make sure to supply 5V to the circuit & 3.3V to AD Converter Vref')

    def calibrate(self) -> None:
        """
        Depreciated, Implemented by Kandai:

        print('start calibrating')
        for i in range(n_loops):
            data = self.read()
            for j in range(3):
                self.mins[j] = min(self.mins[j], data[j])
                self.maxs[j] = max(self.maxs[j], data[j])
            print(self.mins, self.maxs)

        self.bias = (self.mins + self.maxs) / 2
        self.scale = 3 / (self.maxs - self.bias)
        self.calibrated = True
        """
        """
        Trying to implement the calibration code according to a pdf sent by Kandai
        """
        print('start calibrating')
        while True:
            try:
                data = self.read()
                # Update for x_min and x_max
                if data[0] < self.x_min:
                    self.x_min = data[0]
                    print("New X min: ", self.x_min)
                elif data[0] > self.x_max:
                    self.x_max = data[0]
                    print("New X max: ", self.x_max)
                # Update for y_min and y_max
                if data[0] < self.y_min:
                    self.y_min = data[0]
                    print("New Y min: ", self.y_min)
                elif data[0] > self.y_max:
                    self.y_max = data[0]
                    print("New Y max: ", self.y_max)
                # Update for z_min and z_max
                if data[0] < self.z_min:
                    self.z_min = data[0]
                    print("New Z min: ", self.z_min)
                elif data[0] > self.z_max:
                    self.z_max = data[0]
                    print("New Z max: ", self.z_max)
            except KeyboardInterrupt:
                # A keyboard interrupt has been issued, I guess I can stop now
                # And print the min max values
                # Reason to implement this is that, you never know when you have to stop
                # and you just can't calibrate your sensors in some 10000 iterations
                break

        print("Min max values stored, Calibration complete")
        # Set calibrated = true for future reference
        self.calibrated = True
        # Congrats, you successfully did the calibration
        # Print all values for your eye candy
        print("X min: ", self.x_min)
        print("X max: ", self.x_max)
        print("Y min: ", self.y_min)
        print("Y max: ", self.y_max)
        print("Z min: ", self.z_min)
        print("Z max: ", self.z_max)

    def _map(self, x: float, raw_min: float, raw_max: float) -> float:
        """
        Re-maps a number from one range to another. That is, a value of fromLow would get mapped to toLow,
        a value of fromHigh to toHigh, values in-between to values in-between, etc.
        Copied directly from: https://www.arduino.cc/reference/en/language/functions/math/map/
        """
        return (x - raw_min) * (self.out_max - self.out_min) / (raw_max - raw_min) + self.out_min

    def read_raw(self):
        """
        returns python list of fetched sensor data. The data ranges from [0, 4096]
        It's basically an ADC output. As MPC3208 is a 12 bit ADC, 2^12 = 4096, hence the 4096 is the
        upper limit
        """
        return [self.adc.read(pin) for pin in self.pins]

    def read(self):
        # Read raw ADC values from the sensor
        data = np.array(self.read_raw())
        # How this equation came up, look for sphinx documentation. There is a more detailed explanation
        # which can't be fir into comments. So please go look, and lemme know if there is any problem with the logic
        #self.data = ((3 / 4096) * data) - 1.5
        self.data = ((((3.3 / 4096) * data) - 2) * 3.1 ) + 1

        # Returning back the data
        return self.data

    def read_calibrated(self):
        """
        returns numpy list of fetched sensor data. The sensor data is adjusted according to calibration data and then
        returned back to the caller function
        """
        # Read raw ADC values from the sensor
        data = np.array(self.read())
        # Map back values according to calibration
        # For more info, read calibrate function documentation
        for index, value in enumerate(data):
            # Index 0 is for x axis. It's a norm, and circuit is fixed that damn way
            # So I can probably right the number here
            if index == 0:
                raw_min = self.x_min
                raw_max = self.x_max
            # Index 1 is for y axis. It's a norm, and circuit is fixed that damn way
            # So I can probably right the number here
            elif index == 1:
                raw_min = self.y_min
                raw_max = self.y_max
            # Index 2 is for z axis. It's a norm, and circuit is fixed that damn way
            # So I can probably right the number here
            elif index == 2:
                raw_min = self.z_min
                raw_max = self.z_max
            else:
                # Data can have only x, y, z co-ordinates. If there are extra, your code might be hallucinating
                # So look it up homie
                raise ("Data has index greater than 2. Something is seriously wrong. DEBUG IT!!!")
            # Now map the values
            data[index] = self._map(value, raw_min, raw_max)
        # How this equation came up, look for sphinx documentation. There is a more detailed explanation
        # which can't be fir into comments. So please go look, and lemme know if there is any problem with the logic
        self.data = ((6 / 4096) * data) - 3.0
        # Returning back the data
        return self.data


if __name__ == '__main__':
    accel_sensor = ADXL335(xpin=0, ypin=1, zpin=2)
    # accel_sensor.calibrate()

    while True:
        data = accel_sensor.read()
        print("Z Acceleration: ", data[2])
        time.sleep(0.5)

