"""
This will serve as quick example for sending the accelerometer to required ROS core node
By setting all environment variables as clearly explained in my blog
https://krishnachaitanya9.github.io/posts/ros_publish_subscribe/
"""
from robotic_skin.sensor.lsm6ds3_accel import LSM6DS3_acclerometer
import os
from sensor_msgs.msg import Imu
import rospy
import json

# Global Variables
# I know setting them is bad, this is just a clean straight example of a Proof-Of-Concept
# That can be used to understand how things work
# You need to set all of them below, Else script might misbehave
# Also make sure no white spaces in the variables

if __name__ == "__main__":
    # Getting all environment variables from the constants file in the same directory
    current_script_path = os.path.dirname(os.path.realpath(__file__))
    config_file = current_script_path + "/constants.json"
    with open(config_file, 'r') as f:
        data = json.load(f)
    imu_number = data["imu_number"]
    GRAVITATIONAL_CONSTANT = data["GRAVITATIONAL_CONSTANT"]
    RPi_bus_num = data["RPi_bus_num"]
    ros_core_ip = data["ros_core_ip"]  # The ROS Core IP
    # 11311 is the default port, you shouldn't change this unless you know what you are doing
    ros_core_port = data["ros_core_port"]
    RPi_IP = data["RPi_IP"]  # The IP of RPi which will be sending packets
    # First Let's initialize all the environment variables so that ROS doesn't whine about it
    os.environ["ROS_MASTER_URI"] = f'http://{ros_core_ip}:{ros_core_port}'
    os.environ["ROS_IP"] = RPi_IP
    # Okay so now lets initialize the accelerometer and send the packets to ROS Core
    accel = LSM6DS3_acclerometer(bus_num=RPi_bus_num)
    rospy.init_node('talker_' + str(imu_number), anonymous=True)
    pub = rospy.Publisher('/imu_data' + str(imu_number), Imu, queue_size=10)
    r = rospy.Rate(100)
    imu_msg = Imu()
    while not rospy.is_shutdown():
        data0_list = accel.read()
        imu_msg.linear_acceleration.x = data0_list[0] * GRAVITATIONAL_CONSTANT
        imu_msg.linear_acceleration.y = data0_list[1] * GRAVITATIONAL_CONSTANT
        imu_msg.linear_acceleration.z = data0_list[2] * GRAVITATIONAL_CONSTANT
        pub.publish(imu_msg)
        r.sleep()
