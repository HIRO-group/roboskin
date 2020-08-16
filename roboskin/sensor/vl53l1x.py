"""
This is a python class for VL53L1X Proximity Sensor
Datasheet Link: https://www.st.com/resource/en/datasheet/vl53l1x.pdf
This library heavily uses VL53L1X pip application: pip install VL53L1X
"""

import VL53L1X
from roboskin.sensor import Sensor


class VL53L1X_ProximitySensor(Sensor):
    """
    Code for VL53L1X distance sensor class.
    """

    def __init__(self, config_file, range_value=0, timing_budget=33000, inter_measurement_period=33):
        """
        Initialize the VL53L1X sensor, test if the python code can reach it or not, if not throw an exception.
        This class requires the below variables to be set in yaml configuration file:
        RPi_bus_num: The Raspberry Pi I2C bus number
        proximity_i2c_address: The I2C address of the sensor
        Additionally this class extends Sensor, so all sensor's configuration should also be passed to this class
        Parameters
        ----------
        range_value : int
            The proximity sensor has 3 ranges, according to the Python Library:
                None = 0 (Set this if you want to set timing budgets yourself)
                SHORT = 1
                MEDIUM = 2
                LONG = 3
            Link: https://pypi.org/project/VL53L1X/
            By default it's kept to long range
        timing_budget : int
            Timing budget in microseconds. # noqa: W291
            A higher timing budget results in greater measurement accuracy, but also a higher power consumption.
        inter_measurement_period : int
            Inter measurement period in milliseconds.
            The inter measurement period must be >= the timing budget, otherwise it will be double the expected value.
        """
        super(VL53L1X_ProximitySensor, self).__init__(config_file)
        self.tof = VL53L1X.VL53L1X(self.config_dict['RPi_bus_num'], self.config_dict['proximity_i2c_address'])
        self.tof.open()
        if range_value in (0, 1, 2, 3):
            # Either use inbuilt range values provided by the vl53l1x library
            # Or set it to 0 and use your own timing budget values
            if range_value == 0:
                self.tof.start_ranging(range_value)
                self.tof.set_timing(timing_budget, inter_measurement_period)
            else:
                self.tof.start_ranging(range_value)
        else:
            raise Exception("The range value passed is not 1 or 2 or 3")

    def calibrate(self):
        """
        This is the calibration function.
        # TODO: Decide whether you have to implement it or not
        Returns
        -------
        None
        """

    def _read_raw(self):
        """
        This is a function which reads the raw values from the sensor, and gives them back to you, unchanged

        Returns
        -------
        float
            Raw sensor reading from the proximity sensor
        """
        # get_distance get's the distance in mm
        return self.tof.get_distance()

    def _calibrate_values(self, input_value):
        """
        Output the calibrated/corrected value from the input value
        Parameters
        ----------
        input_value : float
            Input value (in "mm") which needs to be calibrated/corrected

        Returns
        -------
        float
            Corrected value from raw value (in "m" A/C to ROS Range standards)
        """
        # To Get distance in metres according to ROS Range msg standards
        # http://docs.ros.org/melodic/api/sensor_msgs/html/msg/Range.html
        return float(input_value / 1000)

    def read(self):
        """
        Reads the sensor values and continuously streams them back to the function whoever called it. This is the
        function you need to put while(True) loop for continuous acquisition of accelerometer values.
        Returns
        -------
        float
            Continuous stream of floats from the proximity sensor

        """
        return self._calibrate_values(self._read_raw())

    def stop(self):
        """
        Stop VL53L1X ToF Sensor Ranging
        """
        self.tof.stop_ranging()
