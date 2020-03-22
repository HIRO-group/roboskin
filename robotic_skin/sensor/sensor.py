"""
Generic Sensor Module
"""
import yaml
import os


class Sensor(object):
    """
    Sensor Class
    """
    def __init__(self, config_file):
        """
        Sensor initialization. This class requires these fields to be set in yaml configuration file for it's working:
        ros_core_ip: The ROS Master IP
        ros_core_port: The port no at which roscore is running (It's usually 11311)
        RPi_IP: The IP of the raspberry PI/PC on which this code is currently running
        Parameters
        ----------
        config_file: str
            The input is full path string from where the class can read the yaml configuration file
        """
        # First Find the dirname, so that we can include some more config files which should included be by default
        config_folder = os.path.dirname(config_file)
        with open(config_file, 'r') as cf:
            # We would be saving everything in a dictionary, so that if there are any duplicates
            # It would raise an error
            self.config_dict = yaml.load(cf)
        # Now we have included all required config files
        # Now set the environment variables
        os.environ["ROS_MASTER_URI"] = 'http://%s:%d' % (self.config_dict['ros_core_ip'],
                                                         self.config_dict['ros_core_port'])
        os.environ["ROS_IP"] = self.config_dict['RPi_IP']

    def calibrate(self):
        """
        Calibrate the sensor
        """
        raise NotImplementedError()

    def read(self):
        """
        Fetch sensor data
        """
        raise NotImplementedError()
