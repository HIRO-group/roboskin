#!/usr/bin/env python

"""
This is a ROS proximity data publisher
"""
import sys
import rospy
from sensor_msgs.msg import Range
import vl53l1x


def publish_proximity(debug=False):
    """
    publish_proximity() function
    """
    rospy.init_node('proximity_publisher', anonymous=True)
    pub = rospy.Publisher('/proximity', Range, queue_size=10)
    rate = rospy.Rate(100)  # Start publishing at 100hz
    ps = vl53l1x.VL53L1X_ProximitySensor()
    range_msg = Range()
    range_msg.radiation_type = 1
    range_msg.min_range = 0
    range_msg.max_range = 3
    # Reference: https://www.st.com/resource/en/datasheet/vl53l1x.pdf
    range_msg.field_of_view = 39.60
    while not rospy.is_shutdown():
        range_msg.range = ps.read()
        if debug:
            print(range_msg)
        pub.publish(range_msg)
        rate.sleep()
    ps.stop()


if __name__ == "__main__":
    try:
        if len(sys.argv) == 2:
            debug = bool(sys.argv[1])
            publish_proximity(debug)
        else:
            publish_proximity()
    except rospy.ROSInterruptException:
        print("Stopped publishing proximity data")
