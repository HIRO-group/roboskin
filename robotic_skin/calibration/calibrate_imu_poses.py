#!/usr/bin/env python
"""
Module for Kinematics Estimation.
"""
import os
import sys
import logging
import argparse
from collections import namedtuple
import pickle
import numpy as np
import rospkg

from robotic_skin.calibration.utils import load_robot_configs
from robotic_skin.calibration.kinematic_chain import KinematicChain
from robotic_skin.calibration import (
    optimizer,
    error_functions,
    stop_conditions,
    loss
)


class KinematicEstimator():
    """
    Class for estimating the kinematics of the arm
    and corresponding sensor unit positions.
    """
    def __init__(self, data, robot_configs, optimizer_function,
                 error_functions_dict, stop_conditions_dict,
                 optimize_all, method_name='OM'):
        """
        Arguments
        ------------
        data: np.ndarray
            Traning data consists of accelerations when in static and dynamic.
            Static accelerations indicate the gravity vector in SU frame.
            Dynamic accelerations indicate the maximum acceleration in SU frame.
            Gravity Vectors are measured at each pose for all accelerometers [pose, accelerometer, xyz].
            Maximum accelerations are measured at each pose excited by each joint
            for all accelerometers [pose, accelerometer, joint].
        """
        # Assume n_sensor is equal to n_joint for now
        self.robot_configs = robot_configs
        self.method_name = method_name
        # We keep these variables just in case
        self.pose_names = list(data.dynamic.keys())
        self.joint_names = list(data.dynamic[self.pose_names[0]].keys())
        self.imu_names = list(data.dynamic[self.pose_names[0]][self.joint_names[0]].keys())
        self.n_pose = len(self.pose_names)
        self.n_joint = len(self.joint_names)
        self.n_sensor = len(self.imu_names)

        # TODO: Clean these shits
        su_joint_dict = {i: i for i in range(self.n_joint)}
        # bounds for DH parameters
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

        if 'dh_parameter' not in robot_configs:
            optimize_all = False

        linkdh_dict = robot_configs['dh_parameter'] if not optimize_all else None
        sudh_dict = None
        # sudh_dict = robot_configs['su_dh_parameter']
        eval_poses = np.array(robot_configs['eval_poses'])

        self.kinematic_chain = KinematicChain(
            n_joint=self.n_joint,
            su_joint_dict=su_joint_dict,
            bound_dict=bound_dict,
            linkdh_dict=linkdh_dict,
            sudh_dict=sudh_dict,
            eval_poses=eval_poses)

        # Below is an example of what error_functions and stop_conditions dictionary looks like
        # error_functions = {
        #     'Rotation': error_func_rotation(data, loss_func()),
        #     'Translation': error_func_transaltion(data, loss_func())}
        # stop_conditions = {
        #     'Rotation': PassThroughStopCondition(),
        #     'Translation': DeltaXStopCondition()}
        self.optimizer = optimizer_function(
            self.kinematic_chain,
            error_functions_dict,
            stop_conditions_dict,
            optimize_all=optimize_all)

        if optimize_all:
            linkdh0 = np.array([0, 0.333, 0, 0])
            su0 = np.random.rand(6)
            params = np.r_[linkdh0, su0]
            self.kinematic_chain.set_params_at(0, params)

        # TODO: Make this a Data Class
        self.all_euclidean_distances = []
        self.estimated_dh_params = []
        self.all_orientations = []
        self.cumulative_data = []

    def optimize(self):
        """
        Optimizes DH parameters using the training data.
        The error function is defined by the error between the measured accelerations
        and estimated accelerations.
        Nonlinear Optimizer will optimize the parameters by decreasing the error.
        There are two different types of error: static and dynamic motion.
        For further explanation, read each function.
        """
        # Optimize each joint (& sensor) at a time from the root
        # currently starting from 6th skin unit

        print('Skipping 0th IMU')
        for i_su in range(1, self.n_sensor):
            print("Optimizing %ith SU ..." % (i_su))

            # optimize parameters wrt data
            params = self.optimizer.optimize(i_su)

            # Compute necessary data
            self.kinematic_chain.set_params_at(i_su, params)
            T = self.kinematic_chain.compute_su_TM(i_su, pose_type='eval')

            euclidean_distance = np.linalg.norm(
                T.position - self.robot_configs['su_pose'][f'su{i_su+1}']['position'])  # noqa: E999

            # Append All the Data to the list
            self.all_euclidean_distances.append(euclidean_distance)
            self.all_orientations.append(T.quaternion)
            self.cumulative_data.append(self.optimizer.all_poses)
            self.estimated_dh_params.append(params)

            print('='*100)
            print('Position:', T.position)
            print('Quaternion:', T.quaternion)
            print('Euclidean distance between real and predicted points: ', euclidean_distance)
            print('='*100)

        self.all_euclidean_distances = np.array(self.all_euclidean_distances)
        self.all_orientations = np.array(self.all_orientations)

        # once done, save to file.
        ros_robotic_skin_path = rospkg.RosPack().get_path('ros_robotic_skin')
        save_path = os.path.join(ros_robotic_skin_path, 'data', f'{self.method_name}_data.pkl')  # noqa: E999
        pickle.dump(self.cumulative_data, open(save_path, "wb"), protocol=2)

        print("Average Euclidean distance = ", np.mean(self.all_euclidean_distances))

    def get_all_accelerometer_positions(self):
        positions = []
        for i_su in range(self.n_sensor):
            T = self.kinematic_chain.compute_su_TM(i_su, pose_type='eval')
            positions.append(T.position)

        return positions


def load_data(robot, directory=None):
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
            # Check if user is running Python2 or Python3 and switch according to that
            if sys.version_info[0] == 2:
                return pickle.load(f)
            else:
                return pickle.load(f, encoding='latin1')

    if directory is None:
        try:
            directory = os.path.join(rospkg.RosPack().get_path('ros_robotic_skin'), 'data')
        except Exception:
            raise FileNotFoundError('ros_robotic_skin not installed in the catkin workspace')

    static = read_pickle('static_data', robot)
    constant = read_pickle('constant_data', robot)
    dynamic = read_pickle('dynamic_data', robot)
    Data = namedtuple('Data', 'static dynamic constant')
    # load all of the data!
    data = Data(static, dynamic, constant)
    # Data = namedtuple('Data', 'static dynamic')
    # data = Data(static, dynamic)

    return data


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

    parser.add_argument('-0', '--optimizer', type=str, default='SeparateOptimizer',
                        help="Please provide an optimizer function for each key provided")
    parser.add_argument('-k', '--all_keys', nargs='+', default=['Rotation', 'Translation'],
                        help="Please Provide a list of keys for the error functions and stop conditions dictionary")
    # parser.add_argument('-e', '--all_error_functions', nargs='+', default=['StaticErrorFunction',
    #                                                                        'ConstantRotationErrorFunction'],
    parser.add_argument('-e', '--all_error_functions', nargs='+', default=['StaticErrorFunction',
                                                                           'MaxAccelerationErrorFunction'],
                        help="Please provide error function for each key provided")
    parser.add_argument('-l', '--all_loss_functions', nargs='+', default=['L2Loss', 'L1Loss'],
                        help="Please provide a loss function for each key provided")
    parser.add_argument('-s', '--stop_conditions', nargs='+', default=['PassThroughStopCondition',
                                                                       'DeltaXStopCondition'],
                        help="Please provide a stop function for each key provided")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    measured_data = load_data(args.robot, args.datadir)
    robot_configs = load_robot_configs(args.configdir, args.robot)

    # Num of dict keys should be all equal
    if not (len(args.all_keys) == len(args.all_error_functions)
            == len(args.all_loss_functions) == len(args.stop_conditions)):
        raise Exception("The # of arguments of all_keys, all_error_functions, all_loss_functions, "
                        "stop_conditions should be same hence exiting...")

    # Select ErrorFunction, StopCondition, Optimizer
    gen_error_functions_dict = {}
    gen_stop_conditions_dict = {}
    for key, error_func, loss_func, stop_func in \
            zip(args.all_keys, args.all_error_functions, args.all_loss_functions, args.stop_conditions):
        error_function = getattr(error_functions, error_func)
        loss_function = getattr(loss, loss_func)
        stop_function = getattr(stop_conditions, stop_func)
        gen_error_functions_dict[key] = error_function(measured_data, loss_function())
        gen_stop_conditions_dict[key] = stop_function()
    optimizer = getattr(optimizer, args.optimizer)

    # log
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.log)
    logging.basicConfig(level=numeric_level)

    # Initialize a Kinematic Estimator
    estimator = KinematicEstimator(measured_data, robot_configs, optimizer,
                                   gen_error_functions_dict, gen_stop_conditions_dict, args.optimizeall)
    # Run Optimization
    estimator.optimize()

    # Get the estimated data
    data = estimator.get_all_accelerometer_positions()
    # Save the data in a file
    ros_robotic_skin_path = rospkg.RosPack().get_path('ros_robotic_skin')
    save_path = os.path.join(ros_robotic_skin_path, 'data', args.savefile)
    np.savetxt(save_path, data)
