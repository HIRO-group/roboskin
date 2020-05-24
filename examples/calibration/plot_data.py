import copy
import argparse
import numpy as np
import matplotlib.pyplot as plt

from robotic_skin.calibration import utils


def is_first(i):
    return i == 0


def is_last(i, n):
    return i == (n - 1)


def plot_side_by_side(y1: np.ndarray, y2: np.ndarray,
                      xlabel: str, title1: str, title2: str):
    if y1.shape != y2.shape:
        raise ValueError('Arguments must be same size')

    n_data = y1.shape[0]
    n_row = y1.shape[1]
    x = np.arange(n_data)

    ylabels = ['ax', 'ay', 'az']
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

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-dd', '--datadir', type=str, default=None,
                        help="Please provide a path to the data directory")
    parser.add_argument('-r', '--robot', type=str, default='panda',
                        help="Currently only 'panda' and 'sawyer' are supported")
    args = parser.parse_args()

    datadir = utils.parse_datadir(args.datadir)
    data, _ = utils.load_data(args.robot, datadir)
    data_noise = copy.deepcopy(data)
    # utils.add_noise(data_noise, 'static', sigma=2)
    # utils.add_noise(data_noise, 'dynamic', sigma=1)
    # utils.add_noise(data_noise, 'constant', sigma=2)
    utils.add_outlier(data_noise, 'static', sigma=2)
    utils.add_outlier(data_noise, 'dynamic', sigma=1)
    utils.add_outlier(data_noise, 'constant', sigma=2)

    n_noise = 10
    noise_sigmas = 0.1 * np.arange(n_noise)

    pose_names = list(data.constant.keys())
    joint_names = list(data.constant[pose_names[0]].keys())
    imu_names = list(data.constant[pose_names[0]][joint_names[0]].keys())

    n_dynamic_pose = len(list(data.dynamic.keys()))
    n_constant_pose = len(list(data.constant.keys()))
    n_static_pose = len(list(data.static.keys()))

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
    plot_side_by_side(
        y1=np.take_along_axis(accelerations, index, axis=0),
        y2=np.take_along_axis(accelerations_noise, index, axis=0),
        xlabel='Data Points',
        title1='Accelerations',
        title2='Accelerations + Noise')

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
    plot_side_by_side(
        y1=np.take_along_axis(accelerations, index, axis=0),
        y2=np.take_along_axis(accelerations_noise, index, axis=0),
        xlabel='Data Points',
        title1='Accelerations',
        title2='Accelerations + Noise')

    for i in range(n_constant_pose):
        for joint in joint_names:
            for imu in imu_names:
                d = data.constant[pose_names[i]][joint][imu][0]
                d_n = data_noise.constant[pose_names[i]][joint][imu][0]
                accelerations = d[:, 4:7]
                accelerations_noise = d_n[:, 4:7]
                plot_side_by_side(
                    y1=accelerations,
                    y2=accelerations_noise,
                    xlabel='Data Points',
                    title1='Accelerations',
                    title2='Accelerations + Noise')
