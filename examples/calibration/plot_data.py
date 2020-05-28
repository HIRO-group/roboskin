import os
import copy
import argparse
import numpy as np
from typing import List
import matplotlib.pyplot as plt

from robotic_skin.calibration.error_functions import estimate_acceleration
from robotic_skin.calibration.kinematic_chain import construct_kinematic_chain
from robotic_skin.calibration import utils

REPODIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIGDIR = os.path.join(REPODIR, 'config')


def is_first(i):
    return i == 0


def is_last(i, n):
    return i == (n - 1)


def plot_side_by_side(y1: np.ndarray, y2: np.ndarray,
                      xlabel: str, title1: str, title2: str, show=True):
    if y1.size == 0 or y2.size == 0:
        print('Data cannot be empty')
        return

    if y1.shape != y2.shape:
        raise ValueError('Arguments must be same size')

    n_data = y1.shape[0]
    n_row = y1.shape[1]
    x = np.arange(n_data)

    ylabels = ['ax', 'ay', 'az'] + ['w', 'alpha']
    fig = plt.figure(figsize=(10, 8))

    for i_row in range(n_row):
        ax_left = fig.add_subplot(n_row, 2, 2*i_row+1)
        ax_right = fig.add_subplot(n_row, 2, 2*i_row+2)

        y = np.hstack((y1[:, i_row], y2[:, i_row]))
        y_min = np.min(y)
        y_max = np.max(y)
        y_range = y_max - y_min
        y_min -= 0.1 * y_range
        y_max += 0.1 * y_range

        ax_left.plot(x, y1[:, i_row])
        ax_left.set_ylabel(ylabels[i_row])
        ax_left.set_ylim([y_min, y_max])

        ax_right.plot(x, y2[:, i_row])
        ax_right.set_ylim([y_min, y_max])

        if is_first(i_row):
            ax_left.set_title(title1)
            ax_right.set_title(title2)
        if is_last(i_row, n_row):
            ax_left.set_xlabel(xlabel)
            ax_right.set_xlabel(xlabel)

    if show:
        plt.show()


def plot_methods_at_once(y_dict: dict, ylabels: List[str], xlabel: str,
                         x: np.ndarray = None, show=True):  # noqa:C901
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

        y = np.array([])
        for method, data in y_dict.items():
            ax.plot(x, data[:, i_row], label=method)
            y = np.hstack((y, data[:, i_row]))

        y_min = np.min(y)
        y_max = np.max(y)
        y_range = y_max - y_min
        y_min -= 0.1 * y_range
        y_max += 0.1 * y_range

        ax.set_ylabel(ylabels[i_row])
        ax.set_ylim([y_min, y_max])

        if is_first(i_row):
            ax.set_title(f'n_data: {n_data}')
            ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                      mode='expand', ncol=len(methods))

        if is_last(i_row, n_row):
            ax.set_xlabel(xlabel)

    if show:
        plt.show()


def verify_if_noise_is_added_correctly(args):
    datadir = utils.parse_datadir(args.datadir)
    data, _ = utils.load_data(args.robot, datadir)
    data_noise = copy.deepcopy(data)

    # Add Noise
    sigma = 1.0
    outlier_ratio = 0.5
    utils.add_outlier(data_noise, 'static', sigma=sigma, outlier_ratio=outlier_ratio)
    utils.add_outlier(data_noise, 'dynamic', sigma=sigma, outlier_ratio=outlier_ratio)
    utils.add_outlier(data_noise, 'constant', sigma=sigma, outlier_ratio=outlier_ratio)

    # Variables
    n_noise = 10

    pose_names = list(data.constant.keys())
    joint_names = list(data.constant[pose_names[0]].keys())
    imu_names = list(data.constant[pose_names[0]][joint_names[0]].keys())

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


def verify_estimated_accelerations_for_dynamic_datacollection(args):
    robot_configs = utils.load_robot_configs(args.configdir, args.robot)

    datadir = utils.parse_datadir(args.datadir)
    data, imu_mappings = utils.load_data(args.robot, datadir)

    kinematic_chain = construct_kinematic_chain(
        robot_configs=robot_configs,
        imu_mappings=imu_mappings,
        test_code=True,
        optimize_all=False)

    pose_names = list(data.dynamic.keys())
    joint_names = list(data.dynamic[pose_names[0]].keys())
    imu_names = list(data.dynamic[pose_names[0]][joint_names[0]].keys())
    print(pose_names, joint_names, imu_names)

    methods = ['analytical', 'mittendorfer', 'normal_mittendorfer']

    for i_su, su in enumerate(imu_names):
        for i_pose, pose in enumerate(pose_names):
            j_joint = kinematic_chain.su_joint_dict[i_su]
            # for i_joint, joint in enumerate(joint_names):
            for i_joint in range(max(0, j_joint-2), j_joint+1):
                joint = joint_names[i_joint]
                d = data.dynamic[pose][joint][su]
                measured_As = d[:, :3]
                joints = d[:, 3:10]
                time = d[:, 10]
                joint_angular_accelerations = d[:, 11]
                max_angular_velocity = d[0, 12]
                joint_angular_velocities = d[:, 13]

                # joint_angular_accelerations = utils.hampel_filter_forloop(
                #     joint_angular_accelerations, 50)[0]
                # joint_angular_accelerations = utils.low_pass_filter(
                #     joint_angular_accelerations, 100, cutoff_freq=10)

                y_dict = {'Measured': measured_As}

                n_data = d.shape[0]
                for method in methods:
                    estimate_As = []
                    additions = []
                    for i_data in range(n_data):
                        kinematic_chain.set_poses(joints[i_data])
                        estimate_A = estimate_acceleration(
                            kinematic_chain=kinematic_chain,
                            i_rotate_joint=i_joint,
                            i_su=i_su,
                            joint_angular_velocity=joint_angular_velocities[i_data],
                            joint_angular_acceleration=joint_angular_accelerations[i_data],
                            max_angular_velocity=max_angular_velocity,
                            current_time=time[i_data],
                            method=method)
                        estimate_As.append(estimate_A)
                        additions.append([joint_angular_velocities[i_data], joint_angular_accelerations[i_data]])
                    estimate_As = np.array(estimate_As)
                    y_dict[method] = estimate_As

                additions = np.array(additions)
                print(np.max(joint_angular_velocities))
                print(f'{su}, {pose}, {joint_names[i_joint]}')
                plot_methods_at_once(
                    y_dict=y_dict,
                    ylabels=['ax', 'ay', 'az'],
                    xlabel='Time [s]',
                    x=time,
                    show=False)

                measured_As = np.hstack((measured_As, additions))
                estimate_As = np.hstack((estimate_As, additions))
                plot_side_by_side(
                    y1=measured_As,
                    y2=estimate_As,
                    xlabel='Time',
                    title1=f'Measured @ [{pose},{joint}{su}]',
                    title2=f'Estimated w/ max_w={max_angular_velocity}'
                )


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

    if args.run == 'verify_noise':
        verify_if_noise_is_added_correctly(args)
    else:
        verify_estimated_accelerations_for_dynamic_datacollection(args)
