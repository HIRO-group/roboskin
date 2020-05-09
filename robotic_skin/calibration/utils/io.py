import os
import yaml
import numpy as np


def n2s(x, precision=2):
    """
    converts numpy array to string.

    Arguments
    ---------
    `x`: `np.array`
        The numpy array to convert to a string.

    `precision`: `int`
        The precision desired on each entry in the array.

    """
    return np.array2string(x, precision=precision, separator=',', suppress_small=True)


def load_robot_configs(configdir, robot):
    """
    Loads robot's DH parameters, SUs' DH parameters and their poses

    configdir: str
        Path to the config directory where robot yaml files exist
    robot: str
        Name of the robot
    """
    filepath = os.path.join(configdir, robot + '.yaml')
    try:
        with open(filepath) as file:
            return yaml.load(file, Loader=yaml.FullLoader)
    except Exception:
        raise ValueError('Please provide a valid config directory with robot yaml files')
