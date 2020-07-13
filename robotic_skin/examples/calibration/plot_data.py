import os
import copy
import argparse
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

import robotic_skin.const as C
from robotic_skin.calibration.error_functions import max_angle_func
from robotic_skin.calibration.kinematic_chain import construct_kinematic_chain
from robotic_skin.calibration.utils.rotational_acceleration import estimate_acceleration
from robotic_skin.calibration import utils

REPODIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
IMGDIR = os.path.join(REPODIR, 'images')
CONFIGDIR = os.path.join(REPODIR, 'config')


def fit_sin(t, y, rotate_joint):
    def optimize_func(x):
        x[0]*np.sin(2*np.pi*C.PATTERN_FREQ[rotate_joint]*t+x[1]) + x[2] - y
    est_amp, est_phase, est_mean = leastsq(optimize_func, [1, 0, 0])[0]
    return est_amp*np.sin(2*np.pi*C.PATTERN_FREQ[rotate_joint]*t+est_phase) + est_mean


def hampel_filter_forloop(input_series, window_size, n_sigmas=3):
    """
    Implementation of Hampel Filter for outlier detection.

    Arguments
    ----------
    `input_series`: `np.array`
        The input data to use for outlier detection.

    `window_size`: `int`
        The sliding window size to use for the filter on
        `input_series`.

    `n_sigmas`: `int`
        The number of standard deviations to determine
        what data points are outliers.
    """
    n = len(input_series)
    new_series = input_series.copy()
    k = 1.4826  # scale factor for Gaussian distribution
    indices = []
    # possibly use np.nanmedian
    for i in range((window_size), (n - window_size)):
        x0 = np.median(input_series[(i - window_size):(i + window_size)])
        S0 = k * np.median(np.abs(input_series[(i - window_size):(i + window_size)] - x0))
        if (np.abs(input_series[i] - x0) > n_sigmas * S0):
            new_series[i] = x0
            indices.append(i)
    return new_series, indices


def low_pass_filter(data, samp_freq, cutoff_freq=15.):
    """
    Implementation of the standard pass filter,
    also known as a exponential moving average filter.

    Arguments
    ---------
    `data`:
        data to be filtered.
    `samp_freq`:
        sampling frequency of the data
    `cutoff_freq`:
        cutoff frequency; that is, data that is > = `cutoff_freq` will
        be attentuated.
    """
    # need to cut cutoff_freq in half because we apply two filters.
    half_cutoff_freq = cutoff_freq * 0.5
    n = len(data)
    # smoother data when alpha is lower
    tau = 1 / (2 * np.pi * half_cutoff_freq)
    dt = 1 / samp_freq
    alpha = dt / (dt + tau)
    new_data = data.copy()

    for i in range(1, n):
        new_data[i] = ((1 - alpha) * new_data[i-1]) + (alpha * data[i])
    reversed_data = new_data[::-1]

    for i in range(1, n):
        reversed_data[i] = ((1 - alpha) * reversed_data[i-1]) + (alpha * reversed_data[i])

    return reversed_data[::-1]


def clean_data(data):
    """
    TO DO - this is hardcoded from ros_robotic_skin, will fix later
    yea....
    """
    data = copy.deepcopy(data)
    for pose_name in data.dynamic.keys():
        for joint_name in data.dynamic[pose_name].keys():
            for imu_name in data.dynamic[pose_name][joint_name].keys():
                # on each pose, for each joint wiggle, get the
                # maximum acceleration for each skin unit
                imu_data = data.dynamic[pose_name][joint_name][imu_name]
                # filter imu acceleration, angular velocities,
                # joint accelerations
                ax = imu_data[:, 0]
                ay = imu_data[:, 1]
                az = imu_data[:, 2]

                joint_accs = imu_data[:, 11]
                if False:
                    pass
                    # use hampel filter for outlier detection
                    # it actually doesn't affect the end result much.
                    ax = hampel_filter_forloop(ax, 10)[0]
                    ay = hampel_filter_forloop(ay, 10)[0]
                    az = hampel_filter_forloop(az, 10)[0]

                    joint_accs = hampel_filter_forloop(joint_accs, 10)[0]

                # filter
                filtered_ax = low_pass_filter(ax, 100.)
                filtered_ay = low_pass_filter(ay, 100.)
                filtered_az = low_pass_filter(az, 100.)

                filtered_joint_accs = low_pass_filter(joint_accs, 100.)

                # array of imu data - both filtered and raw
                data.dynamic[pose_name][joint_name][imu_name][:, 0] = filtered_ax
                data.dynamic[pose_name][joint_name][imu_name][:, 1] = filtered_ay
                data.dynamic[pose_name][joint_name][imu_name][:, 2] = filtered_az
                data.dynamic[pose_name][joint_name][imu_name][:, 11] = filtered_joint_accs

    return data


def check_data(y_dict: dict, ylabels: List[str]):
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

    if len(ylabels) != n_row:
        raise ValueError(f'len of ylabels should be the {n_row}')

    return methods, n_data, n_row


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


def is_left(n_row, n_col, i):
    if i % n_col == 1:
        return True
    return False


def calc_ylim(y, margin=0.1):
    y_min = np.min(y)
    y_max = np.max(y)
    y_range = y_max - y_min
    y_min -= margin * y_range
    y_max += margin * y_range
    return [y_min, y_max]


def ylims_by_rows(y_dict):
    n_row = list(y_dict.values())[0].shape[1]
    ylims = []
    for i_row in range(n_row):
        y = np.array([])
        for each_y in y_dict.values():
            # Compute y_min and y_max for the combined data y
            y = np.hstack((y, each_y[:, i_row]))
        ylim = calc_ylim(y)
        ylims.append(ylim)
    return ylims


def set_captions(ax, xlabel, ylabels, title, ylims, i_row, n_row, n_col):
    ax.set_ylim(ylims[i_row])

    # Set ylabel at the most left
    if is_left(n_row, n_col, n_col*i_row+1):
        ax.set_ylabel(ylabels[i_row])
    # Set Title at the top
    if is_first(i_row):
        ax.set_title(title)
    # Set xlabel at the bottom
    if is_last(i_row, n_row):
        ax.set_xlabel(xlabel)


def savefig(dirname: str, filename: str):
    dirname = os.path.join(IMGDIR, dirname)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    savepath = os.path.join(dirname, filename)
    plt.savefig(savepath)
    print(f'image saved to {savepath}')


def plot_side_by_side(y_dict: dict, title: str,
                      xlabel: str, x: np.ndarray = None,
                      ylabels: List[str] = ['ax', 'ay', 'az'],
                      show=True, save=False):
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
    methods, n_data, n_row, = check_data(y_dict, ylabels)
    n_method = len(methods)
    x = np.arange(n_data) if x is None else x

    # compute ylim
    ylims = ylims_by_rows(y_dict)

    fig = plt.figure(figsize=(10, 8))
    fig.suptitle(title)

    # For All Data/Rows
    for i_col, (method, each_y) in enumerate(y_dict.items()):
        for i_row in range(n_row):
            ax = fig.add_subplot(n_row, n_method, n_method*i_row+i_col+1)
            # Plot Left
            ax.plot(x, each_y[:, i_row])
            set_captions(ax, xlabel, ylabels, method, ylims, i_row, n_row, n_method)

    if show:
        plt.show()

    if save:
        savefig(dirname='plot_side_by_side',
                filename=title+'.png')
    plt.close()


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
    _, n_data, n_row, = check_data(y_dict, ylabels)
    x = np.arange(n_data) if x is None else x

    # compute ylim
    ylims = ylims_by_rows(y_dict)

    fig = plt.figure(figsize=(10, 8))

    for i_row in range(n_row):
        ax = fig.add_subplot(n_row, 1, i_row+1)
        # Plot and combine all data at the same time
        for method, each_y in y_dict.items():
            ax.plot(x, each_y[:, i_row], label=method)

        set_captions(ax, xlabel, ylabels, title, ylims, i_row, n_row, 1)
        ax.legend(loc='best')

    if show:
        plt.show()

    # Save to IMGDIR/plot_in_one_graph directory
    if save:
        savefig(dirname='plot_in_one_graph',
                filename=title+'.png')
    plt.close()


def verify_noise_added_correctly(data, pose_names: List[str],
                                 joint_names: List[str], su_names: List[str],
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
        for su in su_names:
            d = data.static[pose_names[i]][su]
            d_n = data_noise.static[pose_names[i]][su]
            accelerations.append(d[4:7])
            accelerations_noise.append(d_n[4:7])
    accelerations = np.array(accelerations)
    accelerations_noise = np.array(accelerations_noise)
    index = np.argsort(accelerations, axis=0)

    y_dict = {
        'Accelerations': np.take_along_axis(accelerations, index, axis=0),
        'Accelerations + Noise': np.take_along_axis(accelerations_noise, index, axis=0)}
    # Plot
    plot_side_by_side(
        y_dict=y_dict,
        title='Static Acceleration',
        xlabel='Data Points')

    # Plot Dynamic Acceleration Data vs. Noise Added
    # Prepare Data (Append single data points to a list)
    for i in range(n_dynamic_pose):
        for joint in joint_names:
            for su in su_names:
                accelerations = data.dynamic[pose_names[i]][joint][su][:, :3]
                accelerations_noise = data_noise.dynamic[pose_names[i]][joint][su][:, :3]
                y_dict = {
                    'Accelerations': accelerations,
                    'Accelerations + Noise': accelerations_noise}
                # Plot
                plot_side_by_side(
                    y_dict=y_dict,
                    title=f'{joint}_{su}_{pose_names[i]}',
                    xlabel='Data Points',
                    show=False,
                    save=True)

    # Plot Constant Acceleration Data vs. Noise Added
    # Prepare Data
    for i in range(n_constant_pose):
        for joint in joint_names:
            for su in su_names:
                d = data.constant[pose_names[i]][joint][su][0]
                d_n = data_noise.constant[pose_names[i]][joint][su][0]
                accelerations = d[:, 4:7]
                accelerations_noise = d_n[:, 4:7]
                print(f'{pose_names[i]}, {joint}, {su}')
                y_dict = {
                    'Accelerations': accelerations_noise,
                    'Accelerations + Noise': accelerations_noise}
                # Plot
                plot_side_by_side(
                    y_dict=y_dict,
                    title=f'{joint}_{su}_{pose_names[i]}',
                    xlabel='Data Points')


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
    methods = ['analytical', 'our', 'mittendorfer']
    methods = ['our', 'modified_mittendorfer']
    methods = ['our']

    indices = {
        'measured': np.arange(0, 3),
        'joints': np.arange(3, 10),
        'time': 10,
        'angaccel': 11,
        'angvel': 13
    }

    for i_su, su in enumerate(imu_names):
        if i_su != 5:
            continue
        # joint which i_su th SU is attached to
        i_joint = kinematic_chain.su_joint_dict[i_su]
        for pose in pose_names:
            # Consider 2 previous joints
            for rotate_joint in range(max(0, i_joint-2), i_joint+1):
                joint = joint_names[rotate_joint]
                static_acceleration = data.static[pose][su][4:7]
                print(f'[{su}_{pose}_{joint}] {static_acceleration}')

                # Break up the data
                each_data = data.dynamic[pose][joint][su]
                time = each_data[:, indices['time']]

                # Prepare for plotting
                measured_As = each_data[:, indices['measured']]

                ax_fit = fit_sin(time, measured_As[:, 0], rotate_joint)
                ay_fit = fit_sin(time, measured_As[:, 1], rotate_joint)
                az_fit = fit_sin(time, measured_As[:, 2], rotate_joint)
                fit_As = np.array([ax_fit, ay_fit, az_fit]).T

                y_dict = {'Measured': measured_As, 'Fit': fit_As}
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
                    x=time,
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
    data = clean_data(data)

    if args.run == 'verify_noise':
        verify_noise_added_correctly(
            data, pose_names, joint_names, imu_names)
    else:
        verify_acceleration_estimate(
            data, pose_names, joint_names, imu_names,
            robot_configs, imu_mappings)
