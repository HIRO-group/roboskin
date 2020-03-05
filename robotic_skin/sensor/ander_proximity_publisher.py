#!/usr/bin/env python

import rospy
import ander_vl53l1x
from std_msgs.msg import Int16
from sys import argv




def publish_proximity():
    pub = rospy.Publisher('/proximity/y', Int16, queue_size=10)
    rospy.init_node('proximity_publisher', anonymous=True)
    rate = rospy.Rate(100) #10hz
    ps = ander_vl53l1x.VL53L1X_ProximitySensor(range_value=int(argv[1]))

    while not rospy.is_shutdown():
        proximity = ps.read()
        print(proximity, type(proximity))
        pub.publish(proximity)
        rate.sleep()
    ps.stop()

if __name__ == "__main__":
    try: 
        publish_proximity()
    except rospy.ROSInterruptException:
        pass