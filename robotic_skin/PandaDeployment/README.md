# Attaching Accelerometers and Proximity Sensors to Panda
First you make a json file named "deployment_info.json" and add the following info to 
it:
```json
{
  "I2C-0": {
    "imu_id": 1
  },
  "I2C-1": {
    "imu_id": 2
  }
}
```
Where you give Accelerometers an ID which will be unique and shouldn't repeat. I2C-0
the accelerometer connected to I2C-0 of raspberry pi. Similarly I2C-1. To find out which 
I2C port is which you can refer to the RPi pinout here:
https://pi4j.com/1.2/pins/model-3b-rev1.html

Next run the file accelerometer_deployment.py and it should start sending linear acceleration
info to ROS master.