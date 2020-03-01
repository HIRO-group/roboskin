#!/usr/bin/env python
"""
Module for Kinematics Estimation.
"""
import os
import sys
from collections import namedtuple
import pickle
import matplotlib.pyplot as plt
import numpy as np
import nlopt
import rospkg
import pyquaternion as pyqt

import robotic_skin
import robotic_skin.const as C
from robotic_skin.calibration.utils import (
    TransMat, 
    ParameterManager, 
    tfquat_to_pyquat,
    pyquat_to_numpy,
    quaternion_from_two_vectors
)
# Sawyer IMU Position
# THESE ARE THE TRUE VALUES of THE IMU POSITIONS
# IMU0: [0.070, -0.000, 0.160],[-0.000, 0.707, 0.000, 0.707]
# IMU1: [0.086, 0.100, 0.387], [0.024, 0.025, 0.707, 0.707]
# IMU2: [0.324, 0.191, 0.350], [0.012, 0.035, -0.000, 0.999]
# IMU3: [0.485, 0.049, 0.335], [0.045, 0.029, 0.706, 0.706]
# IMU4: [0.709, 0.023, 0.312], [0.008, 0.052, -0.000, 0.999]
# IMU5: [0.883, 0.154, 0.287], [0.045, 0.034, 0.706, 0.706]
# IMU6: [1.087, 0.131, 0.228], [0.489, -0.428, 0.512, 0.562]

# converts numpy array to string
# this function is just for debugging. 
# might be better to fit move to utils file
n2s = lambda x, precision=2 : np.array2string(x, precision=precision, separator=',', suppress_small=True)

def max_acceleration_joint_angle(curr_w, max_w, t):
    """
    max acceleration along a joint angle.
    """
    #th_pattern = np.sign(t) * max_w / (curr_w) * (1 - np.cos(curr_w*t))
    #th_pattern = np.sign(t) * max_w / (2*np.pi*C.PATTERN_FREQ) * (1 - np.cos(2*np.pi*C.PATTERN_FREQ*t))
    th_pattern = max_w / (2*np.pi*C.PATTERN_FREQ) * np.sin(2*np.pi*C.PATTERN_FREQ*t) * t
    #print('-'*20, th_pattern, curr_w, '-'*20)
    return TransMat(th_pattern)

def constant_velocity_joint_angle(curr_w, max_w, t):
    return TransMat(curr_w*t)

class KinematicEstimator():
    """
    Class for estimating the kinematics of the arm
    and sensor unit positions.
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

        self.pose_names = list(data.dynamic.keys())
        self.joint_names = list(data.dynamic[self.pose_names[0]].keys())
        self.imu_names = list(data.dynamic[self.pose_names[0]][self.joint_names[0]].keys())
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
            self.parameter_diffs = np.array([])

            print("Optimizing %ith SU ..."%(i))
            params, bounds = self.param_manager.get_params_at(i=i)
            n_param = params.shape[0]
            Tdofs = self.param_manager.get_tmat_until(i)

            assert len(Tdofs) == i + 1, 'Size of Tdofs supposed to be %i, but %i'%(i+1, len(Tdofs))

            # Construct an global optimizer
            opt = nlopt.opt(C.GLOBAL_OPTIMIZER, n_param)
            # The objective function only accepts x and grad arguments. 
            # This is the only way to pass other arguments to opt
            # https://github.com/JuliaOpt/NLopt.jl/issues/27
            self.previous_params = None 
            opt.set_min_objective(lambda x, grad: self.error_function(x, grad, i, Tdofs))
            # Set boundaries
            opt.set_lower_bounds(bounds[:, 0])
            opt.set_upper_bounds(bounds[:, 1])
            # set stopping threshold
            opt.set_stopval(C.GLOBAL_STOP)
            # opt.set_maxeval(5)
            # Need to set a local optimizer for the global optimizer
            local_opt = nlopt.opt(C.LOCAL_OPTIMIZER, n_param)
            #local_opt.set_stopval(C.LOCAL_STOP)
            # local_opt.set_xtol_abs(1)
            
            opt.set_local_optimizer(local_opt)
            
            # this is where most of the time is spent - in optimization
            params = opt.optimize(params)
            print('='*100)
            # display the parameters
            print('Parameters', n2s(params, 4))
            self.param_manager.set_params_at(i, params)
            pos, quat = self.get_i_accelerometer_position(i)
            print('Position:', pos)
            print('Quaternion:', quat)
            print('='*100)
            # save the optimized parameter to the parameter manager

    def error_function(self, params, grad, i, Tdofs):
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

        ### 0th IMU of panda
        #Tdof2vdof = TransMat(np.array([np.pi/2, -0.147]))
        #Tvdof2su = TransMat(np.array([-np.pi/2, 0.05, 0, np.pi/2]))

        ### 0th IMU fo sawyer
        #Tdof2vdof = TransMat(np.array([np.pi/2, -0.157]))
        #Tvdof2su = TransMat(np.array([-np.pi/2, 0.07, 0, np.pi/2]))

        ### 1th IMU of sawyer
        #Tdof2vdof = TransMat(np.array([-np.pi/2, -0.0925]))
        #Tvdof2su = TransMat(np.array([np.pi/2, 0.07, 0, np.pi/2]))

        Tdof2su = Tdof2vdof.dot(Tvdof2su)
        e1 = self.static_error_function(i, Tdofs, Tdof2su)
        e2 = self.dynamic_error_function(i, Tdofs, Tdof2su)
        e3 = self.rotation_error_function(i, Tdofs, Tdof2su)

        pos, quat = self.get_an_accelerometer_position(Tdofs, Tdof2su)

        if self.previous_params is None:
            self.xdiff = None
            self.previous_params = np.array(params)
        else:
            self.xdiff = np.linalg.norm(np.array(params) - self.previous_params)
            self.previous_params = np.array(params)
        if self.xdiff is not None:
            self.parameter_diffs = np.append(self.parameter_diffs, self.xdiff)
        # get the difference in parameters

        print(n2s(e1+e3, 3), n2s(e1, 5), n2s(e3, 3), n2s(params), n2s(pos), n2s(quat))
        #return e1 + e2
        if len(self.parameter_diffs) >= 10:
            if np.sum(self.parameter_diffs[-11:-1]) <= 0.3:
                return 0.00001
    
        return e1 + e3

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
        
        """
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
        return np.sum(np.linalg.norm(gravities - gravity, axis=1))

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
        gravities = np.zeros((self.n_pose, 3))
        gravity = np.array([0, 0, 9.81])

        errors = 0.0
        for p in range(self.n_pose):
            #for d in range(max(0, i-2), i+1):
            for d in range(i+1):
                data = self.data.constant[self.pose_names[p]][self.joint_names[d]][self.imu_names[i]][0]
                meas_qs = data[:, :4]
                meas_accels = data[:, 4:7] 
                joints = data[:, 7:14] 
                angular_velocities = data[:, 14]

                for meas_q, meas_accel, joint, curr_w in zip(meas_qs, meas_accels, joints, angular_velocities):
                    # Orientation Error
                    model_q = self.estimate_imu_q(Tdofs, Tdof2su, joint[:i+1])
                    meas_q = tfquat_to_pyquat(meas_q)
                    error1 = pyqt.Quaternion.absolute_distance(model_q, meas_q)

                    # Acceleration Error
                    Tjoints = [TransMat(joint) for joint in joint[:i+1]]
                    model_accel = self.estimate_acceleration(Tdofs, Tjoints, Tdof2su, d, curr_w, None, constant_velocity_joint_angle)
                    #error2 = np.sum(np.abs(model_accel - meas_accel))
                    error2 = np.sum(np.linalg.norm(model_accel - meas_accel))
                    """
                    print('Constant [(%.3f,  %.3f, %.3f, %.3f), (%.3f,  %.3f, %.3f, %.3f)]   [(%.3f,  %.3f, %.3f), (%.3f,  %.3f, %.3f)]'%\
                        (model_q[0], model_q[1], model_q[2], model_q[3], meas_q[0], meas_q[1], meas_q[2], meas_q[3], \
                            model_accel[0], model_accel[1], model_accel[2], meas_accel[0], meas_accel[1], meas_accel[2]))
                    """

                    errors += error1 + error2

        return errors

    def estimate_imu_q(self, Tdofs, Tdof2su, joints):
        T = TransMat(np.zeros(4)) 
        for Tdof, j in zip(Tdofs, joints):
            T = T.dot(Tdof).dot(TransMat(j))
        T = T.dot(Tdof2su)

        Rrs2su = T.R.T
        x_rs = np.array([1, 0, 0])
        y_rs = np.array([0, 1, 0])
        z_rs = np.array([0, 0, 1])
        x_su = np.dot(Rrs2su, x_rs)
        y_su = np.dot(Rrs2su, y_rs)
        z_su = np.dot(Rrs2su, z_rs)
        x_su = x_su / np.linalg.norm(x_su)
        y_su = y_su / np.linalg.norm(y_su)
        z_su = z_su / np.linalg.norm(z_su)
        
        q_from_x = quaternion_from_two_vectors(x_rs, x_su)
        q_from_y = quaternion_from_two_vectors(y_rs, y_su)
        q_from_z = quaternion_from_two_vectors(z_rs, z_su)

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
        """
        e2 = 0.0
        for p in range(self.n_pose):
            for d in range(max(0, i-2), i+1):
                max_accel_train = self.data.dynamic[self.pose_names[p]][self.joint_names[d]][self.imu_names[i]][:3]
                curr_w = self.data.dynamic[self.pose_names[p]][self.joint_names[d]][self.imu_names[i]][3]
                max_w = self.data.dynamic[self.pose_names[p]][self.joint_names[d]][self.imu_names[i]][4]
                joints = self.data.dynamic[self.pose_names[p]][self.joint_names[d]][self.imu_names[i]][5:5+i+1]
                Tjoints = [TransMat(joint) for joint in joints]
                max_accel_model = self.estimate_acceleration(Tdofs, Tjoints, Tdof2su, d, curr_w, max_w, max_acceleration_joint_angle)
                # if p == 0:
                #     print('[Dynamic Max Accel, %ith Joint]'%(d), n2s(max_accel_train), n2s(max_accel_model), curr_w, max_w)
                error = np.sum(np.abs(max_accel_train - max_accel_model))
                e2 += error

        return e2

    def estimate_acceleration(self, Tdofs, Tjoints, Tdof2su, d, curr_w, max_w, joint_angle_func):
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
        """
        # Compute Transformation Matrix from RS to SU
        T = TransMat(np.zeros(4)) 
        for Tdof, Tjoint in zip(Tdofs, Tjoints):
            T = T.dot(Tdof).dot(Tjoint)
        T = T.dot(Tdof2su)
        
        Rrs2su = T.R.T

        # Compute Acceleration at RS frame
        dt = 1.0/30.0
        pos = lambda dt: self.accelerometer_position(dt, Tdofs, Tjoints, Tdof2su, d, curr_w, max_w, joint_angle_func)
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
        """
        T = TransMat(np.zeros(4))
        for i_joint, (Tdof, Tjoint) in enumerate(zip(Tdofs, Tjoints)):
            T = T.dot(Tdof).dot(Tjoint)
            if i_joint == d:
                Tpattern = joint_angle_func(curr_w, max_w, t)
                #print(Tpattern.parameters, curr_w, max_w, t, d)
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
        q_from_x = quaternion_from_two_vectors(x_su, x_rs)
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
        q_from_x = quaternion_from_two_vectors(x_rs, x_su)
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

    filename = '_'.join(['dynamic_data', robot])
    filepath = os.path.join(directory, filename + '.pickle')
    with open(filepath, 'rb') as f:
        dynamic = pickle.load(f, encoding='latin1')

    filename = '_'.join(['constant_data', robot])
    filepath = os.path.join(directory, filename + '.pickle')
    with open(filepath, 'rb') as f:
        constant = pickle.load(f, encoding='latin1')

    Data = namedtuple('Data', 'static dynamic constant')
    data = Data(static, dynamic, constant)

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