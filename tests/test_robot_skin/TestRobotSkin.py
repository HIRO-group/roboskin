from robotic_skin.sensor.lsm6ds3_accel import LSM6DS3_acclerometer
from robotic_skin.sensor.vl53l1x import VL53L1X_ProximitySensor

if __name__ == "__main__":
    # Just initializing the sensors, the code will check for reaching them and it can be used as a litmus test
    LSM6DS3_acclerometer()
    VL53L1X_ProximitySensor()