import os
import copy
import numpy as np
import matplotlib.pyplot as plt

from robotic_skin.calibration.optimizer import (
    OurMethodOptimizer,
    MittendorferMethodOptimizer,
)
from robotic_skin.calibration.kinematic_chain import construct_kinematic_chain
from robotic_skin.calibration.data_logger import DataLogger
from robotic_skin.calibration.evaluator import Evaluator
from robotic_skin.calibration.stop_conditions import (
    CombinedStopCondition,
    DeltaXStopCondition,
    MaxCountStopCondition,
)
from robotic_skin.calibration import utils
from calibrate_imu_poses import parse_arguments

REPODIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIGDIR = os.path.join(REPODIR, 'config')


def initialize_optimizers_and_loggers(args, robotic_configs, imu_mappings, datadir, evaluator):
    optimizers = []
    data_loggers = []
    # Our Method
    stop_conditions = {
        'Orientation': DeltaXStopCondition(threshold=0.00001),
        'Position': CombinedStopCondition(
            s1=DeltaXStopCondition(),
            s2=MaxCountStopCondition(count_limit=700)
        )
    }
    kinematic_chain = construct_kinematic_chain(
        robot_configs, imu_mappings, args.test, args.optimizeall)
    data_logger = DataLogger(datadir, args.robot, args.method)
    optimizer = OurMethodOptimizer(
        kinematic_chain, evaluator, data_logger,
        args.optimizeall, args.error_functions, stop_conditions)
    optimizers.append(optimizer)
    data_loggers.append(data_logger)

    kinematic_chain = construct_kinematic_chain(
        robot_configs, imu_mappings, args.test, args.optimizeall)
    data_logger = DataLogger(datadir, args.robot, args.method)
    optimizer = MittendorferMethodOptimizer(
        kinematic_chain, evaluator, data_logger,
        args.optimizeall, args.error_functions, args.stop_conditions, apply_normal_mittendorfer=True)
    optimizers.append(optimizer)
    data_loggers.append(data_logger)

    kinematic_chain = construct_kinematic_chain(
        robot_configs, imu_mappings, args.test, args.optimizeall)
    data_logger = DataLogger(datadir, args.robot, args.method)
    optimizer = MittendorferMethodOptimizer(
        kinematic_chain, evaluator, data_logger,
        args.optimizeall, args.error_functions, args.stop_conditions, apply_normal_mittendorfer=False)
    optimizers.append(optimizer)
    data_loggers.append(data_logger)
    return optimizers, data_loggers


def run_optimizations(measured_data, optimizers, data_loggers, method_names, n_noise=10, sigma=0.1, outlier_ratio=1):
    noise_sigmas = sigma * np.arange(n_noise)
    ave_euclidean_distance = np.zeros((n_noise, len(method_names)))

    for i, noise_sigma in enumerate(noise_sigmas):
        data = copy.deepcopy(measured_data)
        utils.add_outlier(data, 'static', sigma=noise_sigma, outlier_ratio=outlier_ratio)
        utils.add_outlier(data, 'constant', sigma=noise_sigma, outlier_ratio=outlier_ratio)
        utils.add_outlier(data, 'dynamic', sigma=noise_sigma, outlier_ratio=outlier_ratio)
        for j, (optimizer, data_logger) in enumerate(zip(optimizers, data_loggers)):
            optimizer.optimize(data)
            ave_euclidean_distance[i, j] = data_logger.average_euclidean_distance

    print(ave_euclidean_distance)
    return ave_euclidean_distance


def plot_performance(data_logger, method_names, ave_euclidean_distance, n_noise, sigma, filename, title):
    colors = ['-b', '-r', '-g']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    noise = sigma * np.arange(n_noise)
    for i, (data_logger, color, method_name) in enumerate(zip(data_loggers, colors, method_names)):
        ax.plot(noise, ave_euclidean_distance[:, i], color, label=method_name)
    ax.set_title(title)
    ax.set_xlabel("Noise sigma")
    ax.set_ylabel("Ave. L2 Norm")
    ax.legend(loc="upper left")
    ax.plot()
    filename = os.path.join(REPODIR, "images", filename)
    plt.savefig(filename, format="png")


if __name__ == '__main__':
    args = parse_arguments()
    utils.initialize_logging(args.log, args.logfile)

    robot_configs = utils.load_robot_configs(CONFIGDIR, args.robot)
    evaluator = Evaluator(true_su_pose=robot_configs['su_pose'])

    datadir = utils.parse_datadir(args.datadir)
    measured_data, imu_mappings = utils.load_data(args.robot, datadir)

    method_names = ['OM', 'MM', 'mMM']
    outlier_ratios = [0.1, 1.0]
    n_noise = 10
    sigma = 0.1

    for outlier_ratio in outlier_ratios:
        optimizers, data_loggers = initialize_optimizers_and_loggers(
            args,
            robot_configs,
            imu_mappings,
            datadir,
            evaluator
        )

        ave_euclidean_distance = run_optimizations(
            measured_data,
            optimizers,
            data_loggers,
            method_names,
            n_noise=n_noise,
            sigma=sigma,
            outlier_ratio=outlier_ratio
        )

        filename = f"ave_l2_norm_graph_{0}_{(n_noise-1)*sigma}_{outlier_ratio}.png"
        title = f"Ave. L2 norms (sigma={0}-{(n_noise-1)*sigma}, outliers={outlier_ratio})"
        plot_performance(
            data_loggers,
            method_names,
            ave_euclidean_distance,
            n_noise,
            sigma,
            filename,
            title)
