"""
Generic Sensor Module
"""
class Sensor():
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
        