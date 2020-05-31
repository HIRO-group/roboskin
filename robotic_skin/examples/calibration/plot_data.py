import os
import copy
import argparse
import numpy as np
from typing import List
import matplotlib.pyplot as plt

from robotic_skin.calibration.error_functions import max_angle_func
from robotic_skin.calibration.kinematic_chain import construct_kinematic_chain
from robotic_skin.calibration.utils.rotational_acceleration import estimate_acceleration
from robotic_skin.calibration import utils

REPODIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
IMGDIR = os.path.join(REPODIR, 'images')
CONFIGDIR = os.path.join(REPODIR, 'config')


def is_first(i):
    """
    Returns whether it's the first row/column
    """
    return i == 0


def is_last(i, n):
    """
    Returns whether it's the last row/column
    """
    return i == (n - 1)


def plot_side_by_side(y1: np.ndarray, y2: np.ndarray,
                      title1: str, title2: str, xlabel: str,
                      ylabels: List[str] = ['ax', 'ay', 'az'],
                      show=True):
    """
    Plot data side by side: `y1` on the left and `y2` on the right

    Arguments
    ----------
    `y1`: `np.ndarray`
        Data. This function mainly targets time series data
        Shape = (length, data)
    `y2`: `np.ndarray`
        Data. This function mainly targets time series data
        Shape = (length, data)
    `title1`: `str`
        Title of y1
    `title2`: `str`
        Title of y2
    `xlabel`: `str`
        Normally, t's Time [s] or No. Data Points
    `ylabels`: `List[str]`
        Data's label
    `show`: `bool`

    """
    if y1.size == 0 or y2.size == 0:
        print('y1 and y2 cannot be empty')
        return

    if y1.shape != y2.shape:
        raise ValueError('y1 and y2 must be same size')

    n_data = y1.shape[0]
    n_row = y1.shape[1]
    x = np.arange(n_data)

    if len(ylabels) != n_row:
        raise ValueError(f'num of ylabels must be {n_row}')

    fig = plt.figure(figsize=(10, 8))

    # For All Data/Rows
    for i_row in range(n_row):
        ax_left = fig.add_subplot(n_row, 2, 2*i_row+1)
        ax_right = fig.add_subplot(n_row, 2, 2*i_row+2)
        # Compute y_min and y_max for the combined data y
        y = np.hstack((y1[:, i_row], y2[:, i_row]))
        y_min = np.min(y)
        y_max = np.max(y)
        y_range = y_max - y_min
        y_min -= 0.1 * y_range
        y_max += 0.1 * y_range
        # Plot Left
        ax_left.plot(x, y1[:, i_row])
        ax_left.set_ylim([y_min, y_max])
        ax_left.set_ylabel(ylabels[i_row])
        # Plot Right
        ax_right.plot(x, y2[:, i_row])
        ax_right.set_ylim([y_min, y_max])

        # Set Title at the top
        if is_first(i_row):
            ax_left.set_title(title1)
            ax_right.set_title(title2)
        # Set Xlabel at the bottom
        if is_last(i_row, n_row):
            ax_left.set_xlabel(xlabel)
            ax_right.set_xlabel(xlabel)

    if show:
        plt.show()


def plot_in_one_graph(y_dict: dict, ylabels: List[str], xlabel: str,  # noqa:C901
                      title: str, x: np.ndarray = None,
                      show=True, save=False):
    """
    Plot all y_dict data in 1 graph.
    Each y_dict data is assumed to be 2 dimension.
    Shape=(length, data)

    Arguments
    ----------
    `x`: `np.ndarray`
        x axis data
    `xlabel`:
        Normally it's Time [s] or No. Data Points
    `y_dict`: `dict`
        Data stored in a dictionary.
        Keys are the names of the data. (Ex. Method names)
        Values include the actual data.
        This function mainly targets time series data
        y_dict['key'].shape = (length, data)
    `ylabels`: `List[str]`
        Labels of the data
    `title`: `str`
        Title
    `show`: `bool`
        Show plot
    `save`: `bool`
        Save plot
    """
    if not isinstance(y_dict, dict):
        raise ValueError('y_dict be a dictionary')
    if len(y_dict) == 0:
        return

    methods = list(y_dict.keys())
    data = y_dict[methods[0]]

    if not isinstance(data, np.ndarray):
        raise ValueError("y_dict's values must be np.ndarray")

    if data.size == 0:
        print('Data is empty')
        return

    if data.ndim != 2:
        raise ValueError("The dimension of each method's data should be 2")

    n_data = data.shape[0]
    n_row = data.shape[1]
    x = np.arange(n_data) if x is None else x

    if len(ylabels) != n_row:
        raise ValueError(f'len of ylabels should be the {n_row}')

    fig = plt.figure(figsize=(10, 8))

    for i_row in range(n_row):
        ax = fig.add_subplot(n_row, 1, i_row+1)
        # Plot and combine all data at the same time
        y = np.array([])
        for method, data in y_dict.items():
            ax.plot(x, data[:, i_row], label=method)
            y = np.hstack((y, data[:, i_row]))

        # Compute y_min and y_max for the combined data y
        y_min = np.min(y)
        y_max = np.max(y)
        y_range = y_max - y_min
        y_min -= 0.1 * y_range
        y_max += 0.1 * y_range
        # For all Row
        ax.set_ylabel(ylabels[i_row])
        ax.set_ylim([y_min, y_max])
        # Set title at the top
        if is_first(i_row):
            ax.set_title(title)
            ax.legend(loc='best')
        # Set xlable at the bottom
        if is_last(i_row, n_row):
            ax.set_xlabel(xlabel)

    if show:
        plt.show()

    # Save to IMGDIR/plot_in_one_graph directory
    if save:
        dirname = os.path.join(IMGDIR, 'plot_in_one_graph')
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        savepath = os.path.join(dirname, title + '.png')
        plt.savefig(savepath)
        print(f'image saved to {savepath}')

    plt.close()


def verify_noise_added_correctly(data, pose_names: List[str],
                                 joint_names: List[str], imu_names: List[str],
                                 sigma: float = 1.0, outlier_ratio: float = 0.5):

    # Add Noise
    data_types = ['static', 'dynamic', 'constant']
    data_noise = copy.deepcopy(data)
    utils.add_outlier(data_noise, data_types, sigma=sigma, outlier_ratio=outlier_ratio)

    n_dynamic_pose = len(list(data.dynamic.keys()))
    n_constant_pose = len(list(data.constant.keys()))
    n_static_pose = len(list(data.static.keys()))

    # Plot Static Acceleration Data vs. Noise Added
    # Prepare Data (Append single data points to a list)
    accelerations = []
    accelerations_noise = []
    for i in range(n_static_pose):
        for imu in imu_names:
            d = data.static[pose_names[i]][imu]
            d_n = data_noise.static[pose_names[i]][imu]
            accelerations.append(d[4:7])
            accelerations_noise.append(d_n[4:7])
    accelerations = np.array(accelerations)
    accelerations_noise = np.array(accelerations_noise)
    index = np.argsort(accelerations, axis=0)
    # Plot
    plot_side_by_side(
        y1=np.take_along_axis(accelerations, index, axis=0),
        y2=np.take_along_axis(accelerations_noise, index, axis=0),
        xlabel='Data Points',
        title1='Accelerations',
        title2='Accelerations + Noise')

    # Plot Dynamic Acceleration Data vs. Noise Added
    # Prepare Data (Append single data points to a list)
    accelerations = []
    accelerations_noise = []
    for i in range(n_dynamic_pose):
        for joint in joint_names:
            for imu in imu_names:
                d = data.dynamic[pose_names[i]][joint][imu][0]
                d_n = data_noise.dynamic[pose_names[i]][joint][imu][0]
                accelerations.append(d[:3])
                accelerations_noise.append(d_n[:3])
    accelerations = np.array(accelerations)
    accelerations_noise = np.array(accelerations_noise)
    index = np.argsort(accelerations, axis=0)
    # Plot
    plot_side_by_side(
        y1=np.take_along_axis(accelerations, index, axis=0),
        y2=np.take_along_axis(accelerations_noise, index, axis=0),
        xlabel='Data Points',
        title1='Accelerations',
        title2='Accelerations + Noise')

    # Plot Constant Acceleration Data vs. Noise Added
    # Prepare Data
    for i in range(n_constant_pose):
        for joint in joint_names:
            for imu in imu_names:
                d = data.constant[pose_names[i]][joint][imu][0]
                d_n = data_noise.constant[pose_names[i]][joint][imu][0]
                accelerations = d[:, 4:7]
                accelerations_noise = d_n[:, 4:7]
                print(f'{pose_names[i]}, {joint}, {imu}')
                # Plot
                plot_side_by_side(
                    y1=accelerations,
                    y2=accelerations_noise,
                    xlabel='Data Points',
                    title1='Accelerations',
                    title2='Accelerations + Noise')


def verify_acceleration_estimate(data, pose_names: List[str],
                                 joint_names: List[str], imu_names: List[str],
                                 robot_configs: dict, imu_mappings: dict):
    # Initialize KinematicChain
    kinematic_chain = construct_kinematic_chain(
        robot_configs=robot_configs,
        imu_mappings=imu_mappings,
        test_code=True,
        optimize_all=False)

    # Methods to Compare
    methods = ['analytical', 'mittendorfer']

    indices = {
        'measured': np.arange(1, 4),
        'joints': np.arange(3, 11),
        'time': 10,
        'angaccel': 11,
        'angvel': 13
    }

    for i_su, su in enumerate(imu_names):
        # joint which i_su th SU is attached to
        i_joint = kinematic_chain.su_joint_dict[i_su]
        for pose in pose_names:
            # Consider 2 previous joints
            for rotate_joint in range(max(0, i_joint-2), i_joint+1):
                joint = joint_names[rotate_joint]
                print(f'[{su}_{pose}_{joint}]')

                # Break up the data
                each_data = data.dynamic[pose][joint][su]

                # Prepare for plotting
                measured_As = each_data[:, indices['measured']]
                y_dict = {'Measured': measured_As}
                # Run Estimate_Acceleration for each method
                for method in methods:
                    # Store to a dictionary
                    y_dict[method] = estimate_acceleration_batch(
                        kinematic_chain=kinematic_chain,
                        data=each_data,
                        rotate_joint=rotate_joint,
                        i_joint=i_joint,
                        i_su=i_su,
                        inds=indices,
                        method=method)

                # Plot all methods in a same graph
                plot_in_one_graph(
                    y_dict=y_dict,
                    ylabels=['ax', 'ay', 'az'],
                    xlabel='Time [s]',
                    title=f'{joint}_{su}_{pose}',
                    x=each_data[:, indices['time']],
                    show=True,
                    save=False)


def estimate_acceleration_batch(kinematic_chain, data: np.ndarray,
                                rotate_joint: int, i_joint: int,
                                i_su: int, inds: dict, method: str):

    estimate_As = []

    # Go through all the data points
    for d in data:
        # First Set current Pose (joints)
        kinematic_chain.set_poses(d[inds['joints']], end_joint=i_joint)
        # Estimate current acceleration wrt the data
        estimate_A = estimate_acceleration(
            kinematic_chain=kinematic_chain,
            i_rotate_joint=rotate_joint,
            i_su=i_su,
            joint_angular_velocity=d[inds['angvel']],
            joint_angular_acceleration=d[inds['angaccel']],
            current_time=d[inds['time']],
            angle_func=max_angle_func,
            method=method)
        estimate_As.append(estimate_A)

    return np.array(estimate_As)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-dd', '--datadir', type=str, default=None,
                        help="Please provide a path to the data directory")
    parser.add_argument('-r', '--robot', type=str, default='panda',
                        help="Currently only 'panda' and 'sawyer' are supported")
    parser.add_argument('-n', '--run', type=str, default=None,
                        help="Choose which plot to run")
    parser.add_argument('-cd', '--configdir', type=str, default=CONFIGDIR,
                        help="Please provide a path to the config directory")
    args = parser.parse_args()

    # Preparation for the initialization of KinematicChain
    robot_configs = utils.load_robot_configs(args.configdir, args.robot)
    datadir = utils.parse_datadir(args.datadir)
    data, imu_mappings = utils.load_data(args.robot, datadir)

    # Data Keys
    pose_names = list(data.dynamic.keys())
    joint_names = list(data.dynamic[pose_names[0]].keys())
    imu_names = list(data.dynamic[pose_names[0]][joint_names[0]].keys())
    print(pose_names, joint_names, imu_names)

    if args.run == 'verify_noise':
        verify_noise_added_correctly(
            data, pose_names, joint_names, imu_names)
    else:
        verify_acceleration_estimate(
            data, pose_names, joint_names, imu_names,
            robot_configs, imu_mappings)
