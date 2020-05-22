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
from robotic_skin.calibration import utils
from calibrate_imu_poses import parse_arguments

REPODIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIGDIR = os.path.join(REPODIR, 'config')


if __name__ == '__main__':
    args = parse_arguments()

    robot_configs = utils.load_robot_configs(CONFIGDIR, args.robot)
    evaluator = Evaluator(true_su_pose=robot_configs['su_pose'])

    datadir = utils.parse_datadir(args.datadir)
    measured_data, imu_mappings = utils.load_data(args.robot, datadir)

    method_names = ['OM', 'MM', 'mMM']
    optimizers = []
    data_loggers = []

    # Our Method
    kinematic_chain = construct_kinematic_chain(
        robot_configs, imu_mappings, args.test, args.optimizeall)
    data_logger = DataLogger(datadir, args.robot, args.method)
    optimizer = OurMethodOptimizer(
        kinematic_chain, evaluator, data_logger,
        args.optimizeall, args.error_functions, args.stop_conditions)
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

    n_noise = 10
    noise_sigmas = 0.1 * np.arange(n_noise)
    ave_euclidean_distance = np.zeros((n_noise, len(method_names)))

    for i, noise_sigma in enumerate(noise_sigmas):
        data = copy.deepcopy(measured_data)
        utils.add_noise(data, 'static', sigma=noise_sigma)
        utils.add_noise(data, 'constant', sigma=noise_sigma)
        utils.add_noise(data, 'dynamic', sigma=noise_sigma)
        for j, (optimizer, data_logger) in enumerate(zip(optimizers, data_loggers)):
            optimizer.optimize(data)
            ave_euclidean_distance[i, j] = data_logger.average_eulidean_distance

    colors = ['-b', '-r', '-g']
    for i, (data_logger, color, method_name) in enumerate(zip(data_loggers, colors, method_names)):
        plt.plot(ave_euclidean_distance[:, i], color, label=method_name)
    plt.title("Ave. L2 norms of SUs over noise")
    plt.xlabel("Noise sigma")
    plt.ylabel("Ave. L2 Norm")
    plt.legend(loc="upper left")
    plt.xticks(np.arange(n_noise), np.arange(1, n_noise + 1))
    plt.plot()
