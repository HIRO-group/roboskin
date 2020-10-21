#!/usr/bin/env python
"""
Module for Kinematics Estimation.
"""
import os
import argparse

from roboskin.calibration.kinematic_chain import construct_kinematic_chain
from roboskin.calibration.optimizer import choose_optimizer
from roboskin.calibration.data_logger import DataLogger
from roboskin.calibration.evaluator import Evaluator
from roboskin.calibration import utils

REPODIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CONFIGDIR = os.path.join(REPODIR, 'config')



def parse_arguments():
    """
    Parse Arguments
    """
    parser = argparse.ArgumentParser(description='Estimating IMU poses')
    parser.add_argument('-r', '--robot', type=str, default='panda',
                        help="Currently only 'panda' and 'sawyer' are supported")
    parser.add_argument('-dd', '--datadir', type=str, default=None,
                        help="Please provide a path to the data directory")
    parser.add_argument('-cd', '--configdir', type=str, default=CONFIGDIR,
                        help="Please provide a path to the config directory")
    parser.add_argument('-sf', '--savefile', type=str, default='estimate_imu_positions.txt',
                        help="Please Provide a filename for saving estimated IMU poses")
    parser.add_argument('--log', type=str, default='WARNING',
                        help="Please provide a log level")
    parser.add_argument('--logfile', type=str, default=None,
                        help="Please provide a log filename to export")
    parser.add_argument('-oa', '--optimizeall', action='store_true',
                        help="Determines if the optimizer will be run to find all of the dh parameters.")
    parser.add_argument('--test', action='store_true',
                        help="Determines if the true SU DH parameters will be used")
    parser.add_argument('--method', type=str, default='OM',
                        help="Please provide a method name")

    parser.add_argument('-e', '--error_functions', nargs='+', default=None,
                        help="Please provide an error function or a list of error functions")
    parser.add_argument('-l', '--loss_functions', nargs='+', default=None,
                        help="Please provide an loss function or a list of loss functions")
    parser.add_argument('-s', '--stop_conditions', nargs='+', default=None,
                        help="Please provide an stop function or a list of stop functions")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    datadir = utils.parse_datadir(args.datadir)
    is_torch = True if args.method == 'TM' else False
    utils.initialize_logging(args.log, args.logfile)

    robot_configs = utils.load_robot_configs(args.configdir, args.robot)
    measured_data, imu_mappings = utils.load_data(args.robot, datadir)

    # Kinematic Chain of a robot - get torch version depending on method.
    kinematic_chain = construct_kinematic_chain(
        robot_configs, imu_mappings, args.test, args.optimizeall, is_torch=is_torch)

    evaluator = Evaluator(true_su_pose=robot_configs['su_pose'])
    data_logger = DataLogger(datadir, args.robot, args.method)

    # Main Loop
    optimizer = choose_optimizer(
        args=args,
        kinematic_chain=kinematic_chain,
        evaluator=evaluator,
        data_logger=data_logger,
        optimize_all=args.optimizeall)
    optimizer.optimize(measured_data)

    data_logger.save()
    print('Positions')
    print(data_logger.position)
    print('Orientations')
    print(data_logger.orientation)
    print('Euclidean Distance')
    print(data_logger.euclidean_distance)
    print('Quaternion Distance')
    print(data_logger.quaternion_distance)
    print('Ave. Euclidean Distance')
    print(data_logger.average_euclidean_distance)
    print('Elapsed Time')
    print(data_logger.elapsed_times['total'])
