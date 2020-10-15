import time
import unittest
from roboskin.sensor.lsm6ds3 import LSM6DS3_IMU


class LSM6DS3_IMUTest(unittest.TestCase):
    """
    LSM6DS3 sensor test cases.
    """
    def test_constructor(self):
        """
        Test that a read from the lsm6ds3 sensor works.
        """
        _ = LSM6DS3_IMU()
        assert True

    def test_read(self):
        imu = LSM6DS3_IMU()
        for _ in range(5):
            data = imu.read()
            time.sleep(1)
        assert len(data) != 0


if __name__ == '__main__':
    unittest.main()
