import VL53L1X
from robotic_skin.sensor import Sensor

#TODO: Add documentation

class VL53L1X_ProximitySensor(Sensor):
    def __init__(self):
        super().__init__()
        self.tof = VL53L1X.VL53L1X(i2c_bus=1, i2c_address=0x29)
        self.tof.open()
        self.tof.start_ranging(3)
        self.tof.set_timing_budget(140)

    def calibrate(self):
        pass

    def _read_raw(self):
        return self.tof.get_distance()

    def _calibrate_values(self, input_value):
        return input_value

    def read(self):
        return self._calibrate_values(self._read_raw())


if __name__ == "__main__":
    from time import sleep

    ps = VL53L1X_ProximitySensor()

    while True:
        print("Proximity Sensor Readin: ", ps.read())
        sleep(0.02)