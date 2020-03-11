#!/usr/bin/env python

"""
This is a ROS proximity data publisher
"""
import sys
import rospy
from std_msgs.msg import Int16
import vl53l1x


def publish_proximity(debug=False):
    """
    publish_proximity() function
    """
    rospy.init_node('proximity_publisher', anonymous=True)
    pub = rospy.Publisher('/proximity/y', Int16, queue_size=10)
    rate = rospy.Rate(100) #Start publishing at 100hz
    ps = vl53l1x.VL53L1X_ProximitySensor()

    while not rospy.is_shutdown():
        proximity = ps.read()
        if debug:
            print(proximity)
        
        pub.publish(proximity)
        rate.sleep()
    ps.stop()

if __name__ == "__main__":
    try: 
        if len(sys.argv) == 2:
            debug = sys.argv[1]
            publish_proximity(debug)
        else:
            publish_proximity()
    except rospy.ROSInterruptException:
        print("Stopped publishing proximity data")
        