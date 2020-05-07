#!/usr/bin/env python
"""
Module for Kinematics Estimation.
"""
import os
import sys
import argparse
from collections import namedtuple
import pickle
import numpy as np
import rospkg

from robotic_skin.calibration.utils import (
    ParameterManager,
    get_IMU_pose,
    load_robot_configs
)
from robotic_skin.calibration import optimizer
from robotic_skin.calibration import error_functions
# from robotic_skin.calibration.loss import L1Loss
from robotic_skin.calibration import loss

# Sawyer IMU Position
# THESE ARE THE TRUE VALUES of THE IMU POSITIONS
# IMUi: [DH parameters] [XYZ Position] [Quaternion]
# IMU0: [1.57,  -0.157,  -1.57, 0.07, 0,  1.57] [0.070, -0.000, 0.16], [-0.000, 0.707, 0.000, 0.707]
# IMU1: [-1.57, -0.0925, 1.57,  0.07, 0,  1.57] [0.086, 0.100, 0.387], [0.024, 0.025, 0.707, 0.707]
# IMU2: [-1.57, -0.16,   1.57,  0.05, 0,  1.57] [0.324, 0.191, 0.350], [0.012, 0.035, -0.000, 0.999]
# IMU3: [-1.57, 0.0165,  1.57,  0.05, 0,  1.57] [0.485, 0.049, 0.335], [0.045, 0.029, 0.706, 0.706]
# IMU4: [-1.57, -0.17,   1.57,  0.05, 0,  1.57] [0.709, 0.023, 0.312], [0.008, 0.052, -0.000, 0.999]
# IMU5: [-1.57, 0.0053,  1.57,  0.04, 0,  1.57] [0.883, 0.154, 0.287], [0.045, 0.034, 0.706, 0.706]
# IMU6: [0.0,   0.12375, 0.0,   0.03, 0, -1.57] [1.087, 0.131, 0.228], [0.489, -0.428, 0.512, 0.562]

# Panda IMU Position
# IMUi: [DH parameters] [XYZ Position] [Quaternion]
# IMU0: [1.57, -0.15, -1.57, 0.05, 0, 1.57] [0.050, -0.000, 0.183] [0.000, 0.707, -0.000, 0.707]
# IMU1: [1.57, 0.06, -1.57, 0.06, 0, 1.57] [0.060, 0.060, 0.333] [-0.500, 0.500, -0.500, 0.500]
# IMU2: [0, -0.08, 0, 0.05, 0, 1.57] [0.000, -0.050, 0.569] [0.707, 0.000, -0.000, 0.707]
# IMU3: [-1.57, 0.08, 1.57, 0.06, 0, 1.57]  [0.023, -0.080, 0.653] [0.482, -0.482, -0.517, 0.517]
# IMU4: [3.14, -0.1, 3.14, 0.1, 0, 1.57] [0.020, 0.100, 0.938] [-0.706, 0.025, 0.025, 0.707]
# IMU5: [-1.57, 0.03, 1.57, 0.05, 0, 1.57] [-0.023, -0.030, 1.041] [0.482, -0.482, -0.517, 0.517]
# IMU6: [1.57, 0, -1.57, 0.05, 0, 1.57] [0.165, 0.000, 1.028] [0.732, 0.000, 0.682, -0.000]


class KinematicEstimator():
    """
    Class for estimating the kinematics of the arm
    and corresponding sensor unit positions.
    """
    def __init__(self, data, robot_configs, optimizer_function, error_functions_dict, stop_conditions_dict, optimize_all,
                 method_name='OM'):
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

        self.pose_names = list(data.dynamic.keys())
        self.joint_names = list(data.dynamic[self.pose_names[0]].keys())
        self.imu_names = list(data.dynamic[self.pose_names[0]][self.joint_names[0]].keys())
        self.n_pose = len(self.pose_names)
        self.n_joint = len(self.joint_names)
        self.n_sensor = self.n_joint

        self.cumulative_data = []

        print(self.pose_names)
        print(self.joint_names)
        print(self.imu_names)

        assert self.n_joint == 7
        assert self.n_sensor == 7

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

        if 'dh_parameter' not in robot_configs:
            optimize_all = False

        robot_dhparams = robot_configs['dh_parameter'] if not optimize_all else None
        self.param_manager = ParameterManager(self.n_joint, bounds, bounds_su, robot_dhparams)

        # Below is an example of what error_functions and stop_conditions dictionary looks like
        # error_functions = {
        #     'Rotation': error_func_rotation(data, loss_func()),
        #     'Translation': error_func_transaltion(data, loss_func())
        # }
        # stop_conditions = {
        #     'Rotation': PassThroughStopCondition(),
        #     'Translation': DeltaXStopCondition()
        # }
        error_functions = error_functions_dict
        stop_conditions = stop_conditions_dict
        self.optimizer = optimizer_function(error_functions, stop_conditions,
                                            optimize_all=optimize_all)
        # self.optimizer = Optimizer(error_functions, stop_conditions)

        self.imu_true_positions = robot_configs['su_pose']
        self.all_euclidean_distances = []
        self.estimated_dh_params = []
        self.all_orientations = []

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
        for i_imu in range(1, self.n_sensor):
            print("Optimizing %ith SU ..." % (i_imu))
            params, bounds = self.param_manager.get_params_at(i=i_imu)
            Tdofs = self.param_manager.get_tmat_until(i_imu)

            assert len(Tdofs) == i_imu + 1, 'Size of Tdofs supposed to be %i, but %i' % (i_imu+1, len(Tdofs))

            # optimize parameters wrt data
            params = self.optimizer.optimize(i_imu, Tdofs, params, bounds)
            self.cumulative_data.append(self.optimizer.all_poses)
            self.param_manager.set_params_at(i_imu, params)
            pos, quat = self.get_i_accelerometer_position(i_imu)
            self.all_orientations.append(quat)
            euclidean_distance = np.linalg.norm(pos - self.imu_true_positions['su%i' % (i_imu+1)]['position'])
            self.all_euclidean_distances.append(euclidean_distance)
            """
            size of params will be different depending on if we are
            optimizing all or just su dh parameters.
            """
            self.estimated_dh_params.append(params)
            print('='*100)
            print('Position:', pos)
            print('Quaternion:', quat)
            print('Euclidean distance between real and predicted points: ', euclidean_distance)
            print('='*100)
        self.all_orientations = np.array(self.all_orientations)
        all_data = np.array(self.cumulative_data)
        # once done, save to file.
        ros_robotic_skin_path = rospkg.RosPack().get_path('ros_robotic_skin')
        save_path = os.path.join(ros_robotic_skin_path, 'data', f'{self.method_name}_data.npy')
        np.save(save_path, all_data)
        print(all_data.shape)
        print("Average Euclidean distance = ", sum(self.all_euclidean_distances) / len(self.all_euclidean_distances))

    def get_i_accelerometer_position(self, i_sensor):
        """
        gets the ith accelerometer position on the robot.

        Arguments
        ---------
        `i_sensor`: `int`
            IMU number `i_sensor`

        Returns
        -------
        `T.position`: `np.array`
            position of the imu

        `T.q`: `np.array`
            orientation of the imu

        """
        return get_IMU_pose(
            self.param_manager.Tdof2dof[:i_sensor+1],
            self.param_manager.Tdof2vdof[i_sensor].dot(
                self.param_manager.Tvdof2su[i_sensor]
            )
        )

    def get_all_accelerometer_positions(self):
        """
        Returns all accelerometer positions in the initial position

        Returns
        --------
        positions: np.ndarray
            All accelerometer positions
        """
        accelerometer_poses = np.zeros((self.n_sensor, 7))
        for i in range(self.n_sensor):
            position, quaternion = self.get_i_accelerometer_position(i)
            accelerometer_poses[i, :] = np.r_[position, quaternion]

        return accelerometer_poses


def load_data(robot):
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

    try:
        directory = os.path.join(rospkg.RosPack().get_path('ros_robotic_skin'), 'data')
    except Exception:
        print('ros_robotic_skin not installed in the catkin workspace')

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
    parser.add_argument('-sf', '--savefile', type=str, default='estimate_imu_positions.txt',
                        help="Please Provide a filename for saving estimated IMU poses")
    parser.add_argument('-cd', '--configdir', type=str, default=os.path.join(repodir, 'config'))
    parser.add_argument('-k', '--all_keys', nargs='+', default=['Rotation', 'Translation'],
                        help="Please Provide a list of keys for the error functions and stop conditions dictionary")
    parser.add_argument('-e', '--all_error_functions', nargs='+', default=['StaticErrorFunction',
                                                                           'ConstantRotationErrorFunction'],
                        help="Please provide error function for each key provided")
    parser.add_argument('-l', '--all_loss_functions', nargs='+', default=['L2Loss', 'L2Loss'],
                        help="Please provide a loss function for each key provided")
    parser.add_argument('-s', '--stop_conditions', nargs='+', default=['PassThroughStopCondition',
                                                                       'DeltaXStopCondition'],
                        help="Please provide a stop function for each key provided")
    parser.add_argument('-0', '--optimizer', type=str, default='SeparateOptimizer',
                        help="Please provide an optimizer function for each key provided")
    parser.add_argument('-oa', '--optimizeall', action='store_true',
                        help="Determines if the optimizer will be run to find all of the dh parameters.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    measured_data = load_data(args.robot)
    robot_configs = load_robot_configs(args.configdir, args.robot)

    if not (len(args.all_keys) == len(args.all_error_functions)
            == len(args.all_loss_functions) == len(args.stop_conditions)):
        raise Exception("The # of arguments of all_keys, all_error_functions, all_loss_functions, "
                        "stop_conditions should be same hence exiting...")
    gen_error_functions_dict = {}
    gen_stop_conditions_dict = {}
    for key, error_func, loss_func, stop_func in \
            zip(args.all_keys, args.all_error_functions, args.all_loss_functions, args.stop_conditions):
        error_function = getattr(error_functions, error_func)
        loss_function = getattr(loss, loss_func)
        stop_function = getattr(optimizer, stop_func)
        gen_error_functions_dict[key] = error_function(measured_data, loss_function())
        gen_stop_conditions_dict[key] = stop_function()
    optimizer = getattr(optimizer, args.optimizer)
    estimator = KinematicEstimator(measured_data, robot_configs, optimizer,
                                   gen_error_functions_dict, gen_stop_conditions_dict, args.optimizeall)

    estimator.optimize()
    data = estimator.get_all_accelerometer_positions()
    ros_robotic_skin_path = rospkg.RosPack().get_path('ros_robotic_skin')
    save_path = os.path.join(ros_robotic_skin_path, 'data', args.savefile)
    np.savetxt(save_path, data)
