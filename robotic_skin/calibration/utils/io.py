import os
import yaml
import pickle
import rospkg
import logging
import numpy as np
from collections import namedtuple


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


def initialize_logging(log_level: str):
    """
    Initialize Logging Module with given log_lvel
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % log_level)
    logging.basicConfig(level=numeric_level)


def parse_datadir(datadir):
    if datadir is None:
        try:
            datadir = os.path.join(rospkg.RosPack().get_path('ros_robotic_skin'), 'data')
        except Exception:
            raise FileNotFoundError('ros_robotic_skin not installed in the catkin workspace or pass --datadir=PATH_TO_DATA_DIRECTORY')

    return datadir


def load_data(robot, directory):
    """
    Function for collecting acceleration data with poses

    Returns
    --------
    data: Data
        Data includes static and dynamic accelerations data
    """
    def read_pickle(filename, robot):
        filename = '_'.join([filename, robot])
        filepath = os.path.join(directory, filename + '.pickle')
        with open(filepath, 'rb') as f:
            return pickle.load(f, encoding='latin1')

    static = read_pickle('static_data', robot)
    constant = read_pickle('constant_data', robot)
    dynamic = read_pickle('dynamic_data', robot)
    Data = namedtuple('Data', 'static dynamic constant')
    # load all of the data!
    data = Data(static, dynamic, constant)

    filepath = os.path.join(directory, 'imu_mappings.pickle')
    with open(filepath, 'rb') as f:
        imu_mappings = pickle.load(f, encoding='latin1')

    return data, imu_mappings


def add_noise(data, data_type: str, sigma=1):
    if data_type not in ['static', 'constant', 'dynamic']:
        raise ValueError('There is no such data_type='+data_type)

    d = getattr(data, data_type)
    imu_indices = {
        'static': [4, 5, 6],
        'constant': [4, 5, 6],
        'dynamic': [1, 2, 3]}
    imu_index = imu_indices[data_type]

    pose_names = list(d.keys())
    joint_names = list(d[pose_names[0]].keys())
    imu_names = list(d[pose_names[0]][joint_names[0]].keys())

    for pose in pose_names:
        for joint in joint_names:
            for imu in imu_names:
                print(data_type, d[pose][joint][imu][0].shape, imu_index)
                d[pose][joint][imu][0][:, imu_index] = np.random.uniform(d[pose][joint][imu][0][:, imu_index], sigma)


def add_outlier(data, data_type: str, sigma=3, outlier_ratio=0.25):
    if data_type not in ['static', 'constant', 'dynamic']:
        raise ValueError('There is no such data_type='+data_type)
    if not (0 <= outlier_ratio <= 1):
        raise ValueError('Outlier Ratio must be between 0 and 1')

    # IMU index differs betw. data_types
    d = getattr(data, data_type)
    imu_indices = {
        'static': [4, 5, 6],
        'constant': [4, 5, 6],
        'dynamic': [1, 2, 3]}
    imu_index = imu_indices[data_type]

    pose_names = list(d.keys())
    joint_names = list(d[pose_names[0]].keys())
    imu_names = list(d[pose_names[0]][joint_names[0]].keys())
    # Add outliers
    for pose in pose_names:
        for joint in joint_names:
            for imu in imu_names:
                n_data = d[pose][joint][imu].shape[0]
                # generate indices to add outliers
                if n_data == 1:
                    index = 0
                else:
                    index = np.random.choice(np.arange(n_data), size=int(n_data*outlier_ratio))
                d[pose][joint][imu][index, imu_index] = np.random.uniform(d[pose][joint][imu][index, imu_index], sigma)
