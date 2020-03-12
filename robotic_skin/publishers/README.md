# Accelerometer Publisher
The accelerometer publisher will get accelerometer data and then send it to ROS core.
The sample constants.json file which you should have in your directory so that the
script reads all the constants is:

```json
{
  "imu_number" : 0,
  "GRAVITATIONAL_CONSTANT" : 9.819,
  "RPi_bus_num" : 1,
  "ros_core_ip" : "192.168.50.118",
  "ros_core_port" : 11311,
  "RPi_IP" : "192.168.50.31"
}
```