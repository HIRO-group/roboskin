#!/usr/bin/env python
"""
Module for Kinematics Estimation.
"""
import os
import pickle
import argparse
import numpy as np
import pyquaternion as pyqt
from datetime import datetime

from robotic_skin.calibration.kinematic_chain import KinematicChain
from robotic_skin.calibration.optimizer import choose_optimizer
from robotic_skin.calibration import utils


class DataLogger():
    def __init__(self, savedir, robot, method):
        self.date = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = robot + '_' + method + '.pickle'
        self.savepath = os.path.join(savedir, filepath)
        self.best_data = {}
        self.trials = {}

    def add_best(self, i_su, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, np.ndarray):
                value = value.tolist()

            if key not in self.best_data:
                self.best_data[key] = {}

            self.best_data[key][i_su] = value

    def add_trial(self, global_step, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, np.ndarray):
                value = value.tolist()

            if global_step not in self.best_data:
                self.best_data[global_step] = {}

            self.best_data[global_step][key] = value

    def save(self):
        data = {
            'date': self.date,
            'best_data': self.best_data,
            'trials': self.trials}
        with open(self.savepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def print(self):
        print('Estimated SU Positions')
        for i, values in self.best_data['position'].items():
            print(f'SU{i}: {utils.n2s(np.array(values), 3)}')

        print('Estimated SU Orientations')
        for i, values in self.best_data['orientation'].items():
            print(f'SU{i}: {utils.n2s(np.array(values), 3)}')


class Evaluator():
    def __init__(self, true_su_pose):
        self.true_su_pose = true_su_pose

    def evaluate(self, T, i_su):
        euclidean_distance = np.linalg.norm(
            T.position - self.true_su_pose[f'su{i_su+1}']['position'])

        q_su = self.true_su_pose[f'su{i_su+1}']['rotation']
        quaternion_distance = pyqt.Quaternion.absolute_distance(
            T.q, utils.np_to_pyqt(q_su))

        return {'position': euclidean_distance,
                'orientation': quaternion_distance}


def construct_kinematic_chain(robot_configs: dict, imu_mappings: dict,
                              test_code=False, optimize_all=False):
    su_joint_dict = {}
    joints = []
    for imu_str, link_str in imu_mappings.items():
        su_joint_dict[int(imu_str[-1])] = int(link_str[-1]) - 1
        joints.append(int(link_str[-1]) - 1)
    joints = np.unique(joints)

    bounds = np.array([
        [0.0, 0.00001],     # th
        [-1.0, 1.0],        # d
        [-0.2, 0.2],        # a     (radius)
        [-np.pi, np.pi]])   # alpha
    bounds_su = np.array([
        [-np.pi, np.pi],    # th
        [-1.0, 1.0],        # d
        [-np.pi, np.pi],    # th
        [0.0, 0.2],         # d
        [0.0, 0.0001],      # a     # 0 gives error
        [0, np.pi]])        # alpha
    bound_dict = {'link': bounds, 'su': bounds_su}

    keys = ['dh_parameter', 'su_dh_parameter', 'eval_poses']
    for key in keys:
        if key not in robot_configs:
            raise KeyError(f'Keys {keys} should exist in robot yaml file')

    linkdh_dict = None if optimize_all else robot_configs['dh_parameter']
    sudh_dict = robot_configs['su_dh_parameter'] if test_code else None
    eval_poses = np.array(robot_configs['eval_poses'])

    kinematic_chain = KinematicChain(
        n_joint=joints.size,
        su_joint_dict=su_joint_dict,
        bound_dict=bound_dict,
        linkdh_dict=linkdh_dict,
        sudh_dict=sudh_dict,
        eval_poses=eval_poses)

    if optimize_all:
        linkdh0 = np.array(linkdh_dict['joint1'])
        su0 = np.random.rand(6)
        params = np.r_[linkdh0, su0]
        kinematic_chain.set_params_at(0, params)

    return kinematic_chain


def parse_arguments():
    """
    Parse Arguments
    """
    repodir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser = argparse.ArgumentParser(description='Estimating IMU poses')
    parser.add_argument('-r', '--robot', type=str, default='panda',
                        help="Currently only 'panda' and 'sawyer' are supported")
    parser.add_argument('-dd', '--datadir', type=str, default=None,
                        help="Please provide a path to the data directory")
    parser.add_argument('-cd', '--configdir', type=str, default=os.path.join(repodir, 'config'),
                        help="Please provide a path to the config directory")
    parser.add_argument('-sf', '--savefile', type=str, default='estimate_imu_positions.txt',
                        help="Please Provide a filename for saving estimated IMU poses")
    parser.add_argument('--log', type=str, default='WARNING',
                        help="Please provide a log level")
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

    utils.initialize_logging(args.log)

    robot_configs = utils.load_robot_configs(args.configdir, args.robot)
    measured_data, imu_mappings = utils.load_data(args.robot, datadir)

    # Kinematic Chain of a robot
    kinematic_chain = construct_kinematic_chain(
        robot_configs, imu_mappings, args.test, args.optimizeall)

    evaluator = Evaluator(true_su_pose=robot_configs['su_pose'])
    data_logger = DataLogger(datadir, args.robot, args.method)

    # Main Loop
    optimizer = choose_optimizer(
        args=args,
        kinematic_chain=kinematic_chain,
        evaluator=evaluator,
        data_logger=data_logger,
        data=measured_data,
        optimize_all=args.optimizeall)
    optimizer.optimize()

    data_logger.save()
    data_logger.print()
