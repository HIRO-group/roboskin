from robotic_skin.sensor.lsm6ds3_accel import LSM6DS3_acclerometer
from robotic_skin.sensor.vl53l1x import VL53L1X_ProximitySensor
import unittest


class TestCircuit(unittest.TestCase):
    def test_circuit_connections(self):
        LSM6DS3_acclerometer()
        VL53L1X_ProximitySensor()


if __name__ == "__main__":
    unittest.main()
