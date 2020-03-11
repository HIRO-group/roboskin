from robotic_skin.sensor.lsm6ds3_accel import LSM6DS3_acclerometer
import json
import os
from sensor_msgs.msg import Imu
import rospy

if __name__ == "__main__":
    current_script_path = os.path.dirname(os.path.realpath(__file__))
    config_file = current_script_path + "/deployment_info.json"
    with open(config_file, 'r') as f:
        data = json.load(f)
    i2c_0_id = data["I2C-0"]["imu_id"]
    i2c_1_id = data["I2C-1"]["imu_id"]
    accel_0 = LSM6DS3_acclerometer(bus_num=0)
    accel_1 = LSM6DS3_acclerometer(bus_num=1)
    rospy.init_node('talker_' + str(i2c_0_id) + "_" + str(i2c_1_id), anonymous=True)
    pub0 = rospy.Publisher('/imu_data' + str(i2c_0_id), Imu, queue_size=10)
    pub1 = rospy.Publisher('/imu_data' + str(i2c_1_id), Imu, queue_size=10)
    r = rospy.Rate(100)
    imu_msg0 = Imu()
    imu_msg1 = Imu()
    GRAVITATIONAL_CONSTANT = 9.819
    while not rospy.is_shutdown():
        data0_list = accel_0.read()
        data1_list = accel_1.read()
        imu_msg0.linear_acceleration.x = data0_list[0] * GRAVITATIONAL_CONSTANT
        imu_msg0.linear_acceleration.y = data0_list[1] * GRAVITATIONAL_CONSTANT
        imu_msg0.linear_acceleration.z = data0_list[2] * GRAVITATIONAL_CONSTANT
        pub0.publish(imu_msg0)
        imu_msg1.linear_acceleration.x = data1_list[0] * GRAVITATIONAL_CONSTANT
        imu_msg1.linear_acceleration.y = data1_list[1] * GRAVITATIONAL_CONSTANT
        imu_msg1.linear_acceleration.z = data1_list[2] * GRAVITATIONAL_CONSTANT
        pub1.publish(imu_msg0)
        r.sleep()
