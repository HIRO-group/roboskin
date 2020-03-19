"""
Generic Sensor Module
"""
import yaml
import os


def merge_two_dicts(x, y):
    """
    This function merges two texts. If there is a duplicate key it raises an error. The reason for raising such error
    is that to let users know that they have same keys in two file config files that they want to use. It's better to
    have only one key in one file rather than different keys in different files and the user doesn't know which one is
    getting used
    Parameters
    ----------
    x: dict
        First Input Dictionary
    y: dict
        Second Input Dictionary

    Returns
    -------
    dict
        The output dictionary after combining both x and y

    """
    # First check whether they are same keys in these dictionaries or not, else raise an exception
    for key_x, _ in x.items():
        for key_y, _ in y.items():
            if key_x == key_y:
                raise Exception("Same configuration in 2 files with key: %s. Please remove the duplicate keys "
                                "from your config files" % key_x)
    z = x.copy()  # start with x's keys and values
    z.update(y)  # modifies z with y's keys and values & returns None
    return z


class Sensor(object):
    """
    Sensor Class
    """
    def __init__(self, config_file):
        """
        Sensor initialization
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
        #  This list should contain all the files which should be included for sure
        required_config_files = ['params.yaml']
        for each_config in required_config_files:
            with open(config_folder + '/' + each_config, 'r') as cf:
                conf_dict = yaml.load(cf)
            # Append everything to main config dict
            self.config_dict = merge_two_dicts(self.config_dict, conf_dict)
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
