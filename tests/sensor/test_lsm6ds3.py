import unittest
from roboskin.sensor.lsm6ds3 import LSM6DS3_IMU


class LSM6DS3_IMUTest(unittest.TestCase):
    """
    LSM6DS3 sensor test cases.
    """
    def test_read(self):
        """
        Test that a read from the lsm6ds3 sensor works.
        """
        pass


if __name__ == '__main__':
    unittest.main()
