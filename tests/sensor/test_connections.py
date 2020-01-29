"""
If both the tests failed, no sensor is detected
If one of the test failed that means that one of the sensor is not reachable via I2C, depending on error message
If both tests pass, Congrats! Go get a beer!
"""
from robotic_skin.sensor.lsm6ds3_accel import LSM6DS3_acclerometer
from robotic_skin.sensor.vl53l1x import VL53L1X_ProximitySensor
import unittest


class TestCircuit(unittest.TestCase):
    def test_accelerometer_connection(self):
        """
        Testing for Acceleromter Connections
        """
        LSM6DS3_acclerometer()

    def test_proximity_connections(self):
        """
        Testing for Proximity Sensor connections
        """
        VL53L1X_ProximitySensor()


if __name__ == "__main__":
    unittest.main()
