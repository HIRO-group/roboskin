#!/usr/bin/env python
"""
Module for Kinematics Estimation.
"""
import os
import sys
import time
from collections import namedtuple
import pickle
import matplotlib.pyplot as plt
import numpy as np
import nlopt
import rospkg

# import robotic_skin
import robotic_skin.const as C
from robotic_skin.calibration.utils import (
    TransMat,
    ParameterManager,
    pyquat_to_numpy,
    quaternion_from_two_vectors,
    n2s
)
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
# IMU4: [1.57, -0.1, 1.57, 0.1, 0, 1.57] [0.020, 0.100, 0.938] [-0.706, 0.025, 0.025, 0.707]
# IMU5: [-1.57, 0.03, 1.57, 0.05, 0, 1.57] [-0.023, -0.030, 1.041] [0.482, -0.482, -0.517, 0.517]
# IMU6: [1.57, 0, -1.57, 0.05, 0, 1.57] [0.165, 0.000, 1.028] [[0.732, 0.000, 0.682, -0.000]

def max_acceleration_joint_angle(curr_w, max_w, t):
    """
    max acceleration along a joint angle of robot.
    """
    # th_pattern = np.sign(t) * max_w / (curr_w) * (1 - np.cos(curr_w*t))
    # th_pattern = np.sign(t) * max_w / (2*np.pi*C.PATTERN_FREQ) * (1 - np.cos(2*np.pi*C.PATTERN_FREQ*t))
    th_pattern = max_w / (2*np.pi*C.PATTERN_FREQ) * np.sin(2*np.pi*C.PATTERN_FREQ*t) * t
    # print('-'*20, th_pattern, curr_w, '-'*20)
    return TransMat(th_pattern)

def constant_velocity_joint_angle(curr_w, max_w, t):
    return TransMat(curr_w*t)

class KinematicEstimator():
    """
    Class for estimating the kinematics of the arm
    and corresponding sensor unit positions.
    """
    def __init__(self, data, dhparams=None):
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
        self.data = data
        self.dhparams = dhparams
        self.parameter_diffs = np.array([])
        self.diffs_accum = 0

        self.pose_names = list(data.constant.keys())
        self.joint_names = list(data.constant[self.pose_names[0]].keys())
        self.imu_names = list(data.constant[self.pose_names[0]][self.joint_names[0]].keys())
        self.n_pose = len(self.pose_names)
        self.n_joint = len(self.joint_names)
        self.n_sensor = self.n_joint

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
        self.rot_index = [0, 2, 5]
        self.pos_index = [1, 3, 4]
        self.param_manager = ParameterManager(self.n_joint, bounds, bounds_su, dhparams)

        self.previous_params = None

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
        for i in range(1, self.n_sensor):
            print("Optimizing %ith SU ..."%(i))
            params, bounds = self.param_manager.get_params_at(i=i)
            n_param = params.shape[0]
            Tdofs = self.param_manager.get_tmat_until(i)

            assert len(Tdofs) == i + 1, 'Size of Tdofs supposed to be %i, but %i' % (i+1, len(Tdofs))

            #################### First Optimize Rotations ####################
            n_param = int(n_param/2)
            param_rot = params[self.rot_index]
            param_pos = params[self.pos_index]

            self.previous_params = None
            opt = nlopt.opt(C.GLOBAL_OPTIMIZER, n_param)
            opt.set_min_objective(lambda x, grad: self.error_function(x, grad, i, Tdofs, param_pos, 'rot'))
            opt.set_lower_bounds(bounds[self.rot_index, 0])
            opt.set_upper_bounds(bounds[self.rot_index, 1])
            opt.set_stopval(C.ROT_GLOBAL_STOP)
            local_opt = nlopt.opt(C.LOCAL_OPTIMIZER, n_param)
            opt.set_local_optimizer(local_opt)
            param_rot = opt.optimize(param_rot)
            print(param_rot)

            self.previous_params = None
            self.parameter_diffs = np.array([])
            opt = nlopt.opt(C.GLOBAL_OPTIMIZER, n_param)
            opt.set_min_objective(lambda x, grad: self.error_function(x, grad, i, Tdofs, param_rot, 'pos'))
            opt.set_lower_bounds(bounds[self.pos_index, 0])
            opt.set_upper_bounds(bounds[self.pos_index, 1])
            opt.set_stopval(C.POS_GLOBAL_STOP)
            local_opt = nlopt.opt(C.LOCAL_OPTIMIZER, n_param)
            opt.set_local_optimizer(local_opt)
            param_pos = opt.optimize(param_pos)
            print(param_pos)

            params[self.rot_index] = param_rot
            params[self.pos_index] = param_pos
            print('='*100)
            # display the parameters
            print('Parameters', n2s(params, 4))

            # ################### Then Optimize for Translations ####################
            self.param_manager.set_params_at(i, params)
            pos, quat = self.get_i_accelerometer_position(i)
            print('Position:', pos)
            print('Quaternion:', quat)
            print('='*100)

            time.sleep(3)
            # save the optimized parameter to the parameter manager

    def error_function(self, target_params, grad, i, Tdofs, const_params, target):
        """
        Computes an error e_T = e_1 + e_2 from current parameters

        Arguments
        ----------
        params: np.ndarray
            Current estimated parameters

        i: int
            ith sensor
        Tdofs: list of TransMat
            Transformation Matrices between Dofs

        grad: np.ndarray
            Gradient, but we do not use any gradient information
            (We could in the future)
        Returns
        ----------
        error: float
            Error between measured values and estimated model outputs
        """
        # For the 1st IMU, we do not need to think of DoF-to-DoF transformation.
        # Thus, 6 Params (2+4 for each IMU).
        # Otherwise 10 = 4 (DoF to DoF) + 6 (IMU)
        # This condition also checks whether DH params are passed or not
        # If given, only the parameters for IMU should be estimated.
        params = np.zeros(6)

        if target == 'rot':
            params[self.rot_index] = target_params
            params[self.pos_index] = const_params
        else:
            params[self.rot_index] = const_params
            params[self.pos_index] = target_params

        if params.shape[0] == 6:
            #     (2)     (4)
            # dof -> vdof -> su
            Tdof2vdof = TransMat(params[:2])
            Tvdof2su = TransMat(params[2:])
        else:
            #     (4)    (2)     (4)
            # dof -> dof -> vdof -> su
            Tdof2vdof = TransMat(params[4:6])
            Tvdof2su = TransMat(params[6:])

        # Tdof2su = Tdof2vdof.dot(Tvdof2su)
        Tdof2su = get_su_transmat(i, 'panda')

        pos, quat = self.get_an_accelerometer_position(Tdofs, Tdof2su)

        if self.previous_params is None:
            self.xdiff = None
            self.previous_params = np.array(target_params)
        else:
            self.xdiff = np.linalg.norm(np.array(target_params) - self.previous_params)
            self.previous_params = np.array(target_params)
        if self.xdiff is not None:
            self.parameter_diffs = np.append(self.parameter_diffs, self.xdiff)

        if target == 'rot':
            e1 = self.static_error_function(i, Tdofs, Tdof2su)
            print('IMU'+str(i), n2s(e1, 5), n2s(params), n2s(pos), n2s(quat))
            # e4 = np.sum(np.abs(params)[[0,2,5]])
            return e1
        else:
            #e2 = self.dynamic_error_function(i, Tdofs, Tdof2su)
            #print(n2s(e2, 5), n2s(params), n2s(pos), n2s(quat))
            #return e2
            e3 = self.rotation_error_function(i, Tdofs, Tdof2su)
            print('IMU'+str(i), n2s(e3, 5), n2s(params), n2s(pos), n2s(quat), self.xdiff)

            if len(self.parameter_diffs) >= 10:
                if np.mean(self.parameter_diffs[-11:-1]) <= 0.001:
                    return 0.00001

            return e3

    def static_error_function(self, i, Tdofs, Tdof2su):
        """
        Computes static error for ith accelerometer.
        Static error is an deviation of the gravity vector for p positions.

        This function implements Equation 15 in the paper.
        .. math:: `e_1 = \Sigma_{p=1}^P |{}^{RS}g_{N,p} - \Sigma_{p=1}^P {}^{RS}g_{N,p}|^2`
        where
        .. math:: `{}^{RS}g_{N,p} = {}^{RS}R_{SU_N}^{mod,p} {}^{SU_N}g_{N,p}`


        Arguments
        ------------
        i: int
            ith sensor
        Tdof2su: TransMat
            Transformation matrix from the last DoF (Virtual DoF) to Sensor Unit
        Tdofs: list of TransMat
            Transformation Matrices between Dofs

        Returns
        --------
        e1: float
            Static Error

        """  # noqa: W605
        gravities = np.zeros((self.n_pose, 3))
        gravity = np.array([0, 0, 9.81])

        for p in range(self.n_pose):
            joints = self.data.static[self.pose_names[p]][self.imu_names[i]][3:3+i+1]
            Tjoints = [TransMat(joint) for joint in joints]

            # 1 Pose are consists for n_joint DoF
            T = TransMat(np.zeros(4))   # equals to I Matrix
            for Tdof, Tjoint in zip(Tdofs, Tjoints):
                T = T.dot(Tdof).dot(Tjoint)
            # DoF to SU
            T = T.dot(Tdof2su)

            Rsu2rs = T.R

            accel_su = self.data.static[self.pose_names[p]][self.imu_names[i]][:3]
            accel_rs = np.dot(Rsu2rs, accel_su)
            gravities[p, :] = accel_rs

        # print('[Static Accel] ', n2s(np.mean(gravities, axis=0), 2), n2s(np.std(gravities, axis=0), 2))

        #return np.sum(np.linalg.norm(gravities - np.mean(gravities, 0), axis=1))
        #return np.sum(np.linalg.norm(gravities - gravity, axis=1))
        return np.mean(np.linalg.norm(gravities - gravity, axis=1))

    def rotation_error_function(self, i, Tdofs, Tdof2su):
        """
        Arguments
        ------------
        i: int
            ith sensor
        Tdof2su: TransMat
            Transformation matrix from the last DoF (Virtual DoF) to Sensor Unit
        Tdofs: list of TransMat
            Transformation Matrices between Dofs

        Returns
        --------
        e1: float
            Static Error
        """
        # to-do: use these variables?
        # gravities = np.zeros((self.n_pose, 3))
        # gravity = np.array([0, 0, 9.81])

        errors = 0.0
        n_data = 0
        for p in range(self.n_pose):
            # for d in range(max(0, i-2), i+1):
            for d in range(i+1):
                data = self.data.constant[self.pose_names[p]][self.joint_names[d]][self.imu_names[i]][0]
                meas_qs = data[:, :4]
                meas_accels = data[:, 4:7]
                joints = data[:, 7:14]
                angular_velocities = data[:, 14]

                for meas_q, meas_accel, joint, curr_w in zip(meas_qs, meas_accels, joints, angular_velocities):
                    # Orientation Error
                    """
                    model_q = self.estimate_imu_q(Tdofs, Tdof2su, joint[:i+1])
                    meas_q = tfquat_to_pyquat(meas_q)
                    error1 = pyqt.Quaternion.absolute_distance(model_q, meas_q)
                    """

                    # Acceleration Error
                    Tjoints = [TransMat(joint) for joint in joint[:i+1]]
                    # model_accel = self.estimate_acceleration_numerically(Tdofs, Tjoints, Tdof2su, d, curr_w, None, constant_velocity_joint_angle)
                    model_accel = self.estimate_acceleration_analytically(Tdofs, Tjoints, Tdof2su, d, i, curr_w)
                    # error2 = np.sum(np.abs(model_accel - meas_accel))
                    error2 = np.sum(np.linalg.norm(model_accel - meas_accel))
                    print(i, d, joint[d], curr_w, n2s(model_accel), n2s(meas_accel))
                    print(n2s(joint))

                    #errors += error1 + error2
                    errors += error2
                    n_data += 1

        return errors/n_data

    def estimate_imu_q(self, Tdofs, Tdof2su, joints):
        T = TransMat(np.zeros(4))
        for Tdof, j in zip(Tdofs, joints):
            T = T.dot(Tdof).dot(TransMat(j))
        T = T.dot(Tdof2su)

        Rrs2su = T.R.T
        x_rs = np.array([1, 0, 0])
        x_su = np.dot(Rrs2su, x_rs)
        x_su = x_su / np.linalg.norm(x_su)
        q_from_x = quaternion_from_two_vectors(from_vec=x_rs, to_vec=x_su)
        #q_from_x = quaternion_from_two_vectors(from_vec=x_su, to_vec=x_rs)

        """
        y_rs = np.array([0, 1, 0])
        z_rs = np.array([0, 0, 1])
        y_su = np.dot(Rrs2su, y_rs)
        z_su = np.dot(Rrs2su, z_rs)
        y_su = y_su / np.linalg.norm(y_su)
        z_su = z_su / np.linalg.norm(z_su)

        q_from_y = quaternion_from_two_vectors(y_rs, y_su)
        q_from_z = quaternion_from_two_vectors(z_rs, z_su)
        """

        # TODO: Find a way to average all the estimated quaternions
        # q = pyqt.Quaternion.average([q_from_x, q_from_y, q_from_z])

        return q_from_x

    def dynamic_error_function(self, i, Tdofs, Tdof2su):
        """
        Compute errors between estimated and measured max acceleration for sensor i

        .. math:: `\Sigma_{p=1}^P\Sigma_{d=i-3, i>0}^i {}^{SU_i}|a_{max}^{model} - a_{max}^{measured}|_{i,d,p}^2`

        Arguments
        ------------
        i: int
            ith sensor
        Tdof2su: TransMat
            Transformation matrix from the last DoF (Virtual DoF) to Sensor Unit
        Tdofs: list of TransMat
            Transformation Matrices between Dofs

        Returns
        --------
        e2: float
            Dynamic Error
        """  # noqa: W605
        e2 = 0.0
        n_data = 0
        for p in range(self.n_pose):
            for d in range(max(0, i-2), i+1):
                max_accel_train = self.data.dynamic[self.pose_names[p]][self.joint_names[d]][self.imu_names[i]][:3]
                curr_w = self.data.dynamic[self.pose_names[p]][self.joint_names[d]][self.imu_names[i]][3]
                # max_w = self.data.dynamic[self.pose_names[p]][self.joint_names[d]][self.imu_names[i]][4]
                joints = self.data.dynamic[self.pose_names[p]][self.joint_names[d]][self.imu_names[i]][5:5+i+1]
                Tjoints = [TransMat(joint) for joint in joints]
                #max_accel_model = self.estimate_acceleration_numerically(Tdofs, Tjoints, Tdof2su, d, curr_w, max_w, max_acceleration_joint_angle)
                max_accel_model = self.estimate_acceleration_analytically(Tdofs, Tjoints, Tdof2su, d, i, curr_w)
                # if p == 0:
                #     print('[Dynamic Max Accel, %ith Joint]'%(d), n2s(max_accel_train), n2s(max_accel_model), curr_w, max_w)
                error = np.sum(np.abs(max_accel_train - max_accel_model))
                e2 += error
                n_data += 1

        return e2/n_data

    def estimate_acceleration_analytically(self, Tdofs, Tjoints, Tdofi2su, d, i, curr_w):
        # Transformation Matrix from su to rs in rs frame
        rs_T_su = TransMat(np.zeros(4))
        # Transformation Matrix from the last DoFi to the excited DoFd
        dofd_T_dofi = TransMat(np.zeros(4))

        for j in range(d+1):
            #print(j)
            rs_T_su = rs_T_su.dot(Tdofs[j]).dot(Tjoints[j])

        for j in range(d+1, i+1):
            #print(j, d, i)
            rs_T_su = rs_T_su.dot(Tdofs[j]).dot(Tjoints[j])
            dofd_T_dofi = dofd_T_dofi.dot(Tdofs[j]).dot(Tjoints[j])

        rs_T_su = rs_T_su.dot(Tdofi2su)
        dof_T_su = dofd_T_dofi.dot(Tdofi2su)

        dofd_r_su = dof_T_su.position
        # Every joint rotates along its own z axis
        w_dofd = np.array([0, 0, curr_w])
        a_dofd = np.dot(w_dofd, np.dot(w_dofd, dofd_r_su))

        g_rs = np.array([0, 0, 9.81])

        a_su = np.dot(dof_T_su.R.T, a_dofd) + np.dot(rs_T_su.R.T, g_rs)

        return a_su


    def estimate_acceleration_numerically(self, Tdofs, Tjoints, Tdof2su, d, curr_w, max_w, joint_angle_func):
        """
        Compute an acceleration value from positions.
        .. math:: `a = \frac{f({\Delta t}) + f({\Delta t) - 2 f(0)}{h^2}`

        This equation came from Taylor Expansion.
        .. math:: f(t+{\Delta t}) = f(t) + hf^{\prime}(t) + \frac{h^2}{2}f^{\prime\prime}(t)
        .. math:: f(t-{\Delta t}) = f(t) - hf^{\prime}(t) + \frac{h^2}{2}f^{\prime\prime}(t)

        Add both equations and plug t=0 to get the above equation

        Arguments
        ------------
        d: int
            dth excited joint
        Tdof2su: TransMat
            Transformation matrix from the last DoF (Virtual DoF) to Sensor Unit
        Tdofs: list of TransMat
            Transformation Matrices between Dofs
        Tjoints: list of TransMat
            Transformation Matrices (Rotation Matrix)

        Returns
        ---------
        acceleration: np.array
            Acceleration computed from positions
        """  # noqa: W605
        # Compute Transformation Matrix from RS to SU
        T = TransMat(np.zeros(4))
        for Tdof, Tjoint in zip(Tdofs, Tjoints):
            T = T.dot(Tdof).dot(Tjoint)
        T = T.dot(Tdof2su)
        Rrs2su = T.R.T

        # Compute Acceleration at RS frame
        dt = 1.0/30.0
        pos = lambda dt: self.accelerometer_position(dt, Tdofs, Tjoints, Tdof2su, d, curr_w, max_w, joint_angle_func)  # noqa: E731
        gravity = np.array([0, 0, 9.81])

        accel_rs = (pos(dt) + pos(-dt) - 2*pos(0)) / (dt**2) + gravity
        accel_su = np.dot(Rrs2su, accel_rs)

        return accel_su

    def accelerometer_position(self, t, Tdofs, Tjoints, Tdof2su, d, curr_w, max_w, joint_angle_func):
        """
        Compute ith accelerometer position excited by joint d in pose p at time t

        At pose p, let o be a joint and x be a sensor unit, then it looks like
                d       i                     dth joint and ith accelerometer

        |o-x-o      -x-o      -x-o      -x-
            \-x-o/    \-x-o/    \-x-o/

        1   2    3    4    5    6    7         th joint
        1    2    3    4    5    6    7      th sensor

        Arguments
        ------------
        d: int
            dth excited joint
        Tdof2su: TransMat
            Transformation matrix from the last DoF (Virtual DoF) to Sensor Unit
        Tdofs: list of TransMat
            Transformation Matrices between Dofs
        Tjoints: list of TransformationMatrix
            Tranformation Matrices of all joints in Pose p
            Tjoint = [T(th_1), T(th_2), ..., T(th_n)] for n joints

        Returns
        ---------
        position: np.array
            Position of the resulting transformation
        """  # noqa: W605
        T = TransMat(np.zeros(4))
        for i_joint, (Tdof, Tjoint) in enumerate(zip(Tdofs, Tjoints)):
            T = T.dot(Tdof).dot(Tjoint)
            if i_joint == d:
                Tpattern = joint_angle_func(curr_w, max_w, t)
                # print(Tpattern.parameters, curr_w, max_w, t, d)
                T = T.dot(Tpattern)

        T = T.dot(Tdof2su)

        return T.position

    def get_tmat(self):
        """
        Returns lists of transformation matrices

        Returns
        -----------
        list of TransMat
            Transformation Matrices between a DoF and its next DoF
        list of TransMat
            Transformation Matrices between a DoF and its virtual DoF
        list of TransMat
            Transformation Matrices between a virtual DoF and its SU
        """
        return self.param_manager.Tdof2dof, \
            self.param_manager.Tdof2vdof, \
            self.param_manager.Tvdof2su

    def get_an_accelerometer_position(self, Tdofs, Tdof2su):
        T = TransMat(np.zeros(4))
        for Tdof in Tdofs:
            T = T.dot(Tdof)

        T = T.dot(Tdof2su)

        x_rs = np.array([1, 0, 0])
        x_su = np.dot(T.R.T, x_rs)
        # rotation matters
        q_from_x = quaternion_from_two_vectors(from_vec=x_rs, to_vec=x_su)
        q = pyquat_to_numpy(q_from_x)

        return T.position, q

    def get_i_accelerometer_position(self, i_sensor):
        T = TransMat(np.zeros(4))
        for j in range(i_sensor+1):
            T = T.dot(self.param_manager.Tdof2dof[j])
        T = T.dot(self.param_manager.Tdof2vdof[i_sensor].dot(self.param_manager.Tvdof2su[i_sensor]))

        x_rs = np.array([1, 0, 0])

        x_su = np.dot(T.R.T, x_rs)
        x_su = x_su / np.linalg.norm(x_su)
        # rotation matters
        q_from_x = quaternion_from_two_vectors(from_vec=x_rs, to_vec=x_su)
        q = pyquat_to_numpy(q_from_x)

        return T.position, q

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
            print(np.r_[position, quaternion])
            accelerometer_poses[i, :] = np.r_[position, quaternion]

        return accelerometer_poses

    def plot_param_diffs(self):
        plt.plot(self.parameter_diffs)
        plt.title("Parameter Differences")
        plt.show()


def get_su_transmat(i, robot):
    if robot == 'saywer':
        params = np.array([
            [1.57,  -0.157,  -1.57, 0.07, 0,  1.57],
            [-1.57, -0.0925, 1.57,  0.07, 0,  1.57],
            [-1.57, -0.16,   1.57,  0.05, 0,  1.57],
            [-1.57, 0.0165,  1.57,  0.05, 0,  1.57],
            [-1.57, -0.17,   1.57,  0.05, 0,  1.57],
            [-1.57, 0.0053,  1.57,  0.04, 0,  1.57],
            [0.0,   0.12375, 0.0,   0.03, 0, -1.57]
        ])
    elif robot == 'panda':
        params = np.array([
            [1.57, -0.15, -1.57, 0.05, 0, 1.57],
            [1.57, 0.06, -1.57, 0.06, 0, 1.57],
            [0, -0.08, 0, 0.05, 0, 1.57],
            [-1.57, 0.08, 1.57, 0.06, 0, 1.57],
            [1.57, -0.1, 1.57, 0.1, 0, 1.57],
            [-1.57, 0.03, 1.57, 0.05, 0, 1.57],
            [1.57, 0, -1.57, 0.05, 0, 1.57]
        ])
    else:
        raise NotImplementedError("Define a robot's DH Parameters")

    Tdof2vdof = TransMat(params[i, :2])
    Tvdof2su = TransMat(params[i, 2:])

    return Tdof2vdof.dot(Tvdof2su)


def load_data(robot):
    """
    Function for collecting acceleration data with poses

    Returns
    --------
    data: Data
        Data includes static and dynamic accelerations data
    """
    directory = os.path.join(rospkg.RosPack().get_path('ros_robotic_skin'), 'data')

    filename = '_'.join(['static_data', robot])
    filepath = os.path.join(directory, filename + '.pickle')
    with open(filepath, 'rb') as f:
        static = pickle.load(f, encoding='latin1')

    # filename = '_'.join(['dynamic_data', robot])
    # filepath = os.path.join(directory, filename + '.pickle')
    # with open(filepath, 'rb') as f:
    #     dynamic = pickle.load(f, encoding='latin1')

    filename = '_'.join(['constant_data', robot])
    filepath = os.path.join(directory, filename + '.pickle')
    with open(filepath, 'rb') as f:
        constant = pickle.load(f, encoding='latin1')

    # Data = namedtuple('Data', 'static dynamic constant')
    # data = Data(static, dynamic, constant)
    Data = namedtuple('Data', 'static constant')
    data = Data(static, constant)

    return data


def load_dhparams(robot):
    # th, d, a, al
    if robot == 'sawyer':
        dhparams = np.array([
            [0,         0.317,      0,      0],
            [np.pi/2,   0.1925,     0.081,  -np.pi/2],
            [0,         0.4,        0,      np.pi/2],
            [0,         -0.1685,    0,      -np.pi/2],
            [0,         0.4,        0,      np.pi/2],
            [0,         0.1363,     0,      -np.pi/2],
            [np.pi,     0.13375,    0,      np.pi/2]
        ])
    else:
        dhparams = np.array([
            [0, 0.333,  0,          0],
            [0, 0,      0,          -np.pi/2],
            [0, 0.316,  0,          np.pi/2],
            [0, 0,      0.0825,     np.pi/2],
            [0, 0.384,  -0.0825,    -np.pi/2],
            [0, 0,      0,          np.pi/2],
            [0, 0,      0.088,      np.pi/2]
        ])

    return dhparams


if __name__ == '__main__':
    # Need data
    robot = sys.argv[1]
    measured_data = load_data(robot)
    dhparams = load_dhparams(robot)

    estimator = KinematicEstimator(measured_data, dhparams)
    estimator.optimize()

    # plot the differences
    estimator.plot_param_diffs()

    data = estimator.get_all_accelerometer_positions()

    save_path = sys.argv[2]
    ros_robotic_skin_path = rospkg.RosPack().get_path('ros_robotic_skin')
    save_path = os.path.join(ros_robotic_skin_path, 'data', save_path)
    np.savetxt(save_path, data)
