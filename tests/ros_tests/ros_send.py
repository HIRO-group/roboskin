#!/usr/bin/env python
# license removed for brevity
import rospy
from robotic_skin.sensor.adxl335 import ADXL335
from sensor_msgs.msg import Imu

def talker():
    pub = rospy.Publisher('/imu', Imu, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    r = rospy.Rate(10)

    accel_sensor = ADXL335(xpin=0, ypin=1, zpin=2)
    imu_msg = Imu()

    while not rospy.is_shutdown():
        data = accel_sensor._read_raw()
        imu_msg.linear_acceleration.x = data[0]
        imu_msg.linear_acceleration.y = data[1]
        imu_msg.linear_acceleration.z = data[2]

        pub.publish(imu_msg)
        r.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException: pass