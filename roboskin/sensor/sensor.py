"""
Generic Sensor Module
"""


class Sensor(object):
    """
    Sensor Class
    """
    def __init__(self):
        """
        Sensor initialization
        """

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
