import os
import yaml
import pickle
import rospkg
import logging
import numpy as np
from collections import namedtuple


def n2s(x, precision=3):
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


def t2s(x):
    """
    the equivalent of printing out
    a pytorch tensor, but in string form.
    """
    return str(x)


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


def initialize_logging(log_level, filename=None):
    """
    Initialize Logging Module with given log_lvel

    Arguments
    ----------
    log_level: str
    filename: str
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % log_level)
    if filename:
        overwrite = 'y'
        if os.path.exists(filename):
            overwrite = input('File {} already exists. Do you want to overwrite [y/n]?'.format(filename))
        if overwrite == 'y':
            logging.basicConfig(level=numeric_level, filename=filename, filemode='w')

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
    dynamic = read_pickle('dynamic_data', robot)
    Data = namedtuple('Data', 'static dynamic')
    # load all of the data!
    data = Data(static, dynamic)

    filepath = os.path.join(directory, 'imu_mappings.pickle')
    with open(filepath, 'rb') as f:
        imu_mappings = pickle.load(f, encoding='latin1')

    return data, imu_mappings


def add_noise(data, data_types, sigma=1):
    """
    Arguments
    ----------
    data_types: List[str]
    """
    return add_outlier(data, data_types, sigma, 1)


def add_outlier(data, data_types, sigma=3, outlier_ratio=0.25):  # noqa:C901
    """
    Arguments
    ----------
    data_types: List[str]
    """
    for data_type in data_types:
        if data_type not in ['static', 'dynamic']:
            raise ValueError('There is no such data_type='+data_type)

        d = getattr(data, data_type)

        imu_indices = {
            'static': [4, 5, 6],
            'dynamic': [0, 1, 2]}
        imu_index = imu_indices[data_type]

        pose_names = list(data.dynamic.keys())
        joint_names = list(data.dynamic[pose_names[0]].keys())
        imu_names = list(data.dynamic[pose_names[0]][joint_names[0]].keys())

        n_dynamic_pose = len(list(data.dynamic.keys()))
        n_static_pose = len(list(data.static.keys()))

        if data_type == 'static':
            for i in range(n_static_pose):
                for imu in imu_names:
                    flag = np.random.choice([0, 1], p=[1-outlier_ratio, outlier_ratio])
                    if not flag:
                        continue
                    d[pose_names[i]][imu][imu_index] += np.random.normal(0, sigma, size=3)
            return

        n_pose = n_dynamic_pose
        for i in range(n_pose):
            for joint in joint_names:
                for imu in imu_names:
                    flag = np.random.choice([0, 1], p=[1-outlier_ratio, outlier_ratio])
                    if not flag:
                        continue
                    shape = d[pose_names[i]][joint][imu][0][imu_index].shape
                    d[pose_names[i]][joint][imu][0][imu_index] += np.random.normal(0, sigma, size=shape)


def outlier_index(data, outlier_ratio):
    n_data = data.shape[0]
    index = np.random.choice(np.arange(n_data), size=int(n_data*outlier_ratio), replace=False)
    if index.size == 0:
        return None
    elif index.size == 1:
        return index[0]

    return index
