#!/usr/bin/env python

"""
This is a ROS proximity data publisher
"""
import rospy
from std_msgs.msg import Int16
import vl53l1x


def publish_proximity():
    """
    publish_proximity() function
    """
    rospy.init_node('proximity_publisher', anonymous=True)
    pub = rospy.Publisher('/proximity/y', Int16, queue_size=10)
    rate = rospy.Rate(100) #10hz
    ps = vl53l1x.VL53L1X_ProximitySensor()

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
        print("Stopped publishing proximity data")
        