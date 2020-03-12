"""
This will serve as quick example for sending the accelerometer to required ROS core node
By setting all environment variables as clearly explained in my blog
https://krishnachaitanya9.github.io/posts/ros_publish_subscribe/
"""
from robotic_skin.sensor.lsm6ds3_accel import LSM6DS3_acclerometer
import os
from sensor_msgs.msg import Imu
import rospy

# Global Variables
# I know setting them is bad, this is just a clean straight example of a Proof-Of-Concept
# That can be used to understand how things work
# You need to set all of them below, Else script might misbehave
# Also make sure no white spaces in the variables
imu_number = 0
GRAVITATIONAL_CONSTANT = 9.819
RPi_bus_num = 1
ros_core_ip = '192.168.50.118'  # The ROS Core IP
ros_core_port = 11311  # This is the default port, you shouldn't change this unless you know what you are doing
RPi_IP = '192.168.50.31'  # The IP of RPi which will be sending packets
# End Global Variables

if __name__ == "__main__":
    # First Let's initialize all the environment variables so that ROS doesn't whine about it
    os.environ["ROS_MASTER_URI"] = 'http://' + ros_core_ip + ':' + str(ros_core_port)
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
