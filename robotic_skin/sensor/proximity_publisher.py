#!/usr/bin/env python

import rospy
import VL53L1X
from robotic_skin.sensor import Sensor
from std_msgs.msg import Int16

class VL53L1X_ProximitySensor(Sensor):
    def __init__(self, i2c_bus=1, i2c_address=0x29, range_value=1):
        """
        Initialize the VL53L1X sensor, test if the python code can reach it or not, if not throw an exception
        Parameters
        ----------
        i2c_bus : int
            This is the bus number. Basically The I2C port number. For our circuit, I connected it I2C port 1,
            So by default it's value I kept as 1. Feel free to pass your own value if you need it.
        i2c_address : int
            (It would be easy for you to pass hexadecimal int of the form 0xNN, directly according to the datasheet)
            The I2C address of the accelerometer. According to the datasheet of VL53L1X the address is 0x29. For future
            note this can be changed to anything if you wish so. This library doesn't handle changing I2C address as of
            now
        range_value : int
            The proximity sensor has 3 ranges, according to the Python Library. This is exactly that int.
            Link: https://pypi.org/project/VL53L1X/
            By default it's kept to long range
        """
        super().__init__()
        self.tof = VL53L1X.VL53L1X(i2c_bus, i2c_address)
        self.tof.open()
        if range_value == 1 or range_value == 2 or range_value == 3:
            self.tof.start_ranging(0)
            self.tof.set_timing(22000, 100)
        else:
            raise("The range value passed is not 1 or 2 or 3")

    def calibrate(self):
        """
        This is the calibration function.
        # TODO: Decide whether you have to implement it or not
        Returns
        -------
        None
        """
        pass

    def _read_raw(self):
        """
        This is a function which reads the raw values from the sensor, and gives them back to you, unchanged

        Returns
        -------
        float
            Raw sensor reading from the proximity sensor
        """
        return self.tof.get_distance()

    def _calibrate_values(self, input_value):
        """
        Output the calibrated/corrected value from the input value
        Parameters
        ----------
        input_value : float

        Returns
        -------
        float
            Corrected value from raw value
        """
        return input_value

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

def publish_proximity():
    pub = rospy.Publisher('/proximity/y', Int16, queue_size=10)
    rospy.init_node('proximity_publisher', anonymous=True)
    rate = rospy.Rate(100) #10hz
    ps = VL53L1X_ProximitySensor()
    
    while not rospy.is_shutdown():
        proximity = ps.read()
        print(proximity, type(proximity))
        pub.publish(proximity)
        rate.sleep()


if __name__ == "__main__":
    try:
        publish_proximity()
    except rospy.ROSInterruptException:
        pass
