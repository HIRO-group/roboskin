"""
Generic Sensor Module
"""
import yaml
import os


def merge_two_dicts(x, y):
    # First check whether they are same keys in these dictionaries or not, else raise an exception
    for key_x, _ in x.items():
        for key_y, _ in y.items():
            if key_x == key_y:
                raise Exception("Same configuration in 2 files with key: %s. Please remove the duplicate keys "
                                "from your config files" % key_x)
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

class Sensor():
    """
    Sensor Class
    """

    def __init__(self, config_file):
        # First Find the dirname, so that we can include some more config files which should be by default
        config_folder = os.path.dirname(config_file)
        #  This list should contain all the files which should be included for sure
        required_config_files = ['params.yaml']
        with open(config_file, 'r') as cf:
            # We would be saving everything in a dictionary, so that if there are any duplicates
            # It would raise an error
            self.config_dict = yaml.load(cf)
        for each_config in required_config_files:
            with open(config_folder+'/'+each_config, 'r') as cf:
                conf_dict = yaml.load(cf)
            # Append everything to main config dict
            self.config_dict = merge_two_dicts(self.config_dict, conf_dict)
        # Now set the environment variables





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

if __name__ == "__main__":
    a = {'a':1, 'b':2}
    b = {'y':3, 't':4}
    c = merge_two_dicts(a, b)
    print(c)
