import os
import copy
import logging
import numpy as np
import matplotlib.pyplot as plt

from robotic_skin.calibration.optimizer import (
    OurMethodOptimizer,
    MittendorferMethodOptimizer,
)
from robotic_skin.calibration.kinematic_chain import construct_kinematic_chain
from robotic_skin.calibration.data_logger import DataLogger
from robotic_skin.calibration.evaluator import Evaluator
from robotic_skin.calibration import utils
from robotic_skin.examples.calibration.calibrate_imu_poses import parse_arguments

REPODIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CONFIGDIR = os.path.join(REPODIR, 'config')


def initialize_optimizers_and_loggers(args, robotic_configs, imu_mappings, datadir, evaluator):
    optimizers = []
    data_loggers = []
    # Our Method
    kinematic_chain = construct_kinematic_chain(
        robot_configs, imu_mappings, args.test, args.optimizeall)
    data_logger = DataLogger(datadir, args.robot, args.method, overwrite=True)
    optimizer = OurMethodOptimizer(
        kinematic_chain, evaluator, data_logger,
        args.optimizeall, args.error_functions, args.stop_conditions)
    optimizers.append(optimizer)
    data_loggers.append(data_logger)

    kinematic_chain = construct_kinematic_chain(
        robot_configs, imu_mappings, args.test, args.optimizeall)
    data_logger = DataLogger(datadir, args.robot, args.method, overwrite=True)
    optimizer = MittendorferMethodOptimizer(
        kinematic_chain, evaluator, data_logger,
        args.optimizeall, args.error_functions, args.stop_conditions, method='mittendorfer')
    optimizers.append(optimizer)
    data_loggers.append(data_logger)

    kinematic_chain = construct_kinematic_chain(
        robot_configs, imu_mappings, args.test, args.optimizeall)
    data_logger = DataLogger(datadir, args.robot, args.method, overwrite=True)
    optimizer = MittendorferMethodOptimizer(
        kinematic_chain, evaluator, data_logger,
        args.optimizeall, args.error_functions, args.stop_conditions, method='modified_mittendorfer')
    optimizers.append(optimizer)
    data_loggers.append(data_logger)
    return optimizers, data_loggers


def run_optimizations(measured_data, optimizers, data_loggers, method_names, n_noise=10, sigma=0.1, outlier_ratio=1):
    noise_sigmas = sigma * np.arange(n_noise)
    ave_euclidean_distance = np.zeros((n_noise, len(method_names)))
    total_time = np.zeros((n_noise, len(method_names)))

    for i, noise_sigma in enumerate(noise_sigmas):
        data = copy.deepcopy(measured_data)
        utils.add_outlier(data, ['static', 'constant', 'dynamic'], sigma=noise_sigma, outlier_ratio=outlier_ratio)
        for j, (optimizer, data_logger) in enumerate(zip(optimizers, data_loggers)):
            logging.info(f'Optimizer: {method_names[j]}, sigma={noise_sigma}, Outlier: {outlier_ratio}')
            optimizer.optimize(data)
            ave_euclidean_distance[i, j] = data_logger.average_euclidean_distance
            total_time[i, j] = data_logger.elapsed_times['total']

    print(ave_euclidean_distance)
    return ave_euclidean_distance, total_time


def plot_performance(data_logger, method_names, colors, ave_euclidean_distance, total_time, n_noise, sigma, filename, title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    noise = sigma * np.arange(n_noise)
    for i, (data_logger, method_name, color) in enumerate(zip(data_loggers, method_names, colors)):
        ax.plot(noise, ave_euclidean_distance[:, i], color, label=method_name)
    ax.set_title(title)
    ax.set_xlabel("Noise sigma")
    ax.set_ylabel("Ave. L2 Norm")
    ax.legend(loc="upper left")
    ax.plot()
    filename = os.path.join(REPODIR, "images", filename)
    plt.savefig(filename, format="png")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    noise = sigma * np.arange(n_noise)
    for i, (data_logger, method_name, color) in enumerate(zip(data_loggers, method_names, colors)):
        ax.plot(noise, total_time[:, i], color, label=method_name)
    ax.set_title(title)
    ax.set_xlabel("Noise sigma")
    ax.set_ylabel("Total Time Spend")
    ax.legend(loc="upper left")
    ax.plot()
    filename = os.path.join(REPODIR, "images", 'time_' + filename)
    plt.savefig(filename, format="png")


if __name__ == '__main__':
    args = parse_arguments()
    utils.initialize_logging(args.log, args.logfile)

    robot_configs = utils.load_robot_configs(CONFIGDIR, args.robot)
    evaluator = Evaluator(true_su_pose=robot_configs['su_pose'])

    datadir = utils.parse_datadir(args.datadir)
    measured_data, imu_mappings = utils.load_data(args.robot, datadir)

    method_names = ['OM', 'MM', 'mMM']
    colors = ['-b', '-r', '-g']
    outlier_ratios = [0.1, 0.5, 1.0]
    sigmas = [1, 0.5, 0.1]
    n_noise = 10

    for outlier_ratio, sigma in zip(outlier_ratios, sigmas):
        optimizers, data_loggers = initialize_optimizers_and_loggers(
            args,
            robot_configs,
            imu_mappings,
            datadir,
            evaluator
        )

        if len(method_names) != len(optimizers):
            raise ValueError('Lengths of method_names and optimizers do not much')

        ave_euclidean_distance, total_time = run_optimizations(
            measured_data,
            optimizers,
            data_loggers,
            method_names,
            n_noise=n_noise,
            sigma=sigma,
            outlier_ratio=outlier_ratio
        )

        filename = f"ave_l2_norm_graph_{0}_{(n_noise-1)*sigma}_{outlier_ratio}_loss1.png"
        title = f"Ave. L2 norms (sigma={0}-{(n_noise-1)*sigma}, outliers={outlier_ratio})"
        plot_performance(
            data_loggers,
            method_names,
            colors,
            ave_euclidean_distance,
            total_time,
            n_noise,
            sigma,
            filename,
            title)
