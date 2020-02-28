#!/usr/bin/env python
"""
Module for Kinematics Estimation
"""
import os
import sys
from collections import namedtuple
import pickle
import numpy as np
import nlopt

import robotic_skin
import robotic_skin.const as C
from robotic_skin.calibration.utils import TransMat, ParameterManager

import rospkg

# need to import set_franka_pose, oscillate_franka

# Sawyer IMU Position
# IMU0: [0.070, -0.000, 0.160]
# IMU1: [0.086, 0.100, 0.387]
# IMU2: [0.324, 0.191, 0.350]
# IMU3: [0.485, 0.049, 0.335]
# IMU4: [0.709, 0.023, 0.312]
# IMU5: [0.883, 0.154, 0.287]
# IMU6: [1.087, 0.131, 0.228]

n2s = lambda x, precision=2 :  np.array2string(x, precision=precision, separator=',', suppress_small=True)

class KinematicEstimator():
    """
    Class for estimating the kinematics of the arm
    and sensor unit positions.
    """
    def __init__(self, data, poses, dhparams=None):
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
        self.poses = poses
        self.dhparams = dhparams
        self.n_pose = poses.shape[0]
        self.n_joint = poses.shape[1]
        self.n_sensor = self.n_joint

        #assert self.n_pose == 20
        assert self.n_joint == 7
        assert self.n_sensor == 7

        self.pose_names = list(data.dynamic.keys())
        self.joint_names = list(data.dynamic[self.pose_names[0]].keys())
        self.imu_names = list(data.dynamic[self.pose_names[0]][self.joint_names[0]].keys())
        print(self.pose_names)
        print(self.joint_names)
        print(self.imu_names)
        
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
        self.param_manager = ParameterManager(self.n_joint, poses, bounds, bounds_su, dhparams)

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
        for i in range(1, self.n_sensor):
            print("Optimizing %ith SU ..."%(i))
            params, bounds = self.param_manager.get_params_at(i=i)
            n_param = params.shape[0]

            Tdofs, Tposes = self.param_manager.get_tmat_until(i)

            assert len(Tdofs) == i + 1, 'Size of Tdofs supposed to be %i, but %i'%(i+1, len(Tdofs))
            assert len(Tposes) == self.n_pose
            for Tpose in Tposes:
                assert len(Tpose) == i + 1

            # Construct an global optimizer
            opt = nlopt.opt(C.GLOBAL_OPTIMIZER, n_param)
            # The objective function only accepts x and grad arguments. 
            # This is the only way to pass other arguments to opt
            # https://github.com/JuliaOpt/NLopt.jl/issues/27
            opt.set_min_objective(lambda x, grad: self.error_function(x, grad, i, Tdofs, Tposes))

            # Set boundaries
            opt.set_lower_bounds(bounds[:, 0])
            opt.set_upper_bounds(bounds[:, 1])
            opt.set_stopval(C.GLOBAL_STOP)
            # Need to set a local optimizer for the global optimizer
            local_opt = nlopt.opt(C.LOCAL_OPTIMIZER, n_param)
            local_opt.set_stopval(C.LOCAL_STOP)
            opt.set_local_optimizer(local_opt)

            params = opt.optimize(params)
            print('='*100)
            print('Parameters', n2s(params, 4))
            pos, vec = self.get_i_accelerometer_position(i)
            print('Position:', pos)
            print(n2s(vec[:,0]))
            print(n2s(vec[:,1]))
            print(n2s(vec[:,2]))
            print('='*100)
            # save the optimized parameter to the parameter manager
            self.param_manager.set_params_at(i, params)

    def error_function(self, params, grad, i, Tdofs, Tposes):
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
        Tposes: list of TransMat
            Transformation Matrices (Rotation Matrix)

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
        e1 = self.static_error_function(i, Tdof2su, Tdofs)
        e2 = self.dynamic_error_function(i, Tdof2su, Tdofs, Tposes)

        pos, _ = self.get_an_accelerometer_position(Tdofs, Tdof2su)

        print(n2s(e1+e2, 3), n2s(e1, 5), n2s(e2, 3), n2s(params), n2s(pos))
        #return e1 + e2
        return e1

    def static_error_function(self, i, Tdof2su, Tdofs):
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
        Tposes: list of TransMat
            Transformation Matrices (Rotation Matrix)

        Returns
        --------
        e1: float
            Static Error
        
        """
        # It should be i DoF + a virtual DoF
        assert len(Tdofs) == i + 1

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

        print('[Static Accel] ', n2s(np.mean(gravities, axis=0), 2), n2s(np.std(gravities, axis=0), 2))

        #return np.sum(np.linalg.norm(gravities - np.mean(gravities, 0), axis=1))
        return np.sum(np.linalg.norm(gravities - gravity, axis=1))

    def dynamic_error_function(self, i, Tdof2su, Tdofs, Tposes):
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
        Tposes: list of TransMat
            Transformation Matrices (Rotation Matrix)

        Returns
        --------
        e2: float
            Dynamic Error
        """
        assert len(Tdofs) == i + 1
        assert len(Tposes) == self.n_pose
        for Tpose in Tposes:
            assert len(Tpose) == i + 1
            
        e2 = 0.0
        for p in range(self.n_pose):
            for d in range(max(0, i-2), i+1):
                max_accel_train = self.data.dynamic[self.pose_names[p]][self.joint_names[d]][self.imu_names[i]][:3]
                max_w = self.data.dynamic[self.pose_names[p]][self.joint_names[d]][self.imu_names[i]][3]
                max_joint_accel = self.data.dynamic[self.pose_names[p]][self.joint_names[d]][self.imu_names[i]][4]
                joints = self.data.dynamic[self.pose_names[p]][self.joint_names[d]][self.imu_names[i]][5:5+i+1]
                Tjoints = [TransMat(joint) for joint in joints]
                max_accel_model = self.estimate_max_acceleration(d, Tdofs, Tjoints, Tdof2su, p, max_w)
                if p == 0:
                    print('[Dynamic Max Accel, %ith Joint]'%(d), n2s(max_accel_train), n2s(max_accel_model), max_w, max_joint_accel, joints[d])
                error = np.sum(np.linalg.norm(max_accel_train - max_accel_model))
                e2 += error

        return e2

    def estimate_max_acceleration(self, d, Tdofs, Tjoints, Tdof2su, p, max_w):
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
        Tposes: list of TransMat
            Transformation Matrices (Rotation Matrix)

        Returns
        ---------
        acceleration: np.array
            Acceleration computed from positions
        """
        # Compute Transformation Matrix from RS to SU at max acceleration
        T = TransMat(np.zeros(4))   # equals to I Matrix
        for Tdof, Tjoint in zip(Tdofs, Tjoints):
            T = T.dot(Tdof).dot(Tjoint)
        T = T.dot(Tdof2su)
        
        Rrs2su = T.R.T

        # Compute Acceleration at RS frame
        dt = 1.0/30.0
        pos = lambda dt: self.accelerometer_position(dt, d, Tdofs, Tjoints, Tdof2su, p, max_w)
        gravity = np.array([0, 0, 9.81])

        accel_rs = (pos(dt) + pos(-dt) - 2*pos(0)) / (dt**2) + gravity
        accel_su = np.dot(Rrs2su, accel_rs)

        return accel_su

    def accelerometer_position(self, t, d, Tdofs, Tjoints, Tdof2su, p, max_w):
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
        # Currently I assume that the patterns are same for all the joints
        # We could change to a list of Tranformation Matrix
        #   Tpatts = [T(th_patt_1), T(th_patt_2), ..., T(th_patt_n)]
        #   Tpatt = Tpatts[d]
        #th_pattern = max_w/(2*np.pi*C.PATTERN_FREQ) * np.sin(2*np.pi*C.PATTERN_FREQ*t)
        th_pattern = max_w/(2*np.pi*C.PATTERN_FREQ) * (1 - np.cos(2*np.pi*C.PATTERN_FREQ*t))
        alpha = max_w*(2*np.pi*C.PATTERN_FREQ) * np.cos(2*np.pi*C.PATTERN_FREQ*t)
        if p == 0 :
            print(alpha)
        Tpatt = TransMat(th_pattern) # for all joint
        
        T = TransMat(np.zeros(4))
        # loop over all joint to get to the ith sensor
        for i_joint, (Tdof, Tjoint) in enumerate(zip(Tdofs, Tjoints)):
            # 1. Transform each joint by dh parameter theta Tdofs[i_joint]
            # 2. Then Rotate the axis by theta_pose Tjoints[i_joint] defined by the pose
            T = T.dot(Tdof).dot(Tjoint)
            # If at the dth joint, Rotate the joint by theta_pattern
            if i_joint == d:
                T = T.dot(Tpatt)

        # At the end, Transform from the last ith DoF to ith SU
        T = T.dot(Tdof2su)
        
        # Return only the XYZ position of the sensor in Reference Frame
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

        I = np.eye(3)
        vectors = np.dot(T.R, I)

        return T.position, vectors

    def get_i_accelerometer_position(self, i_sensor):
        T = TransMat(np.zeros(4))
        for j in range(i_sensor):
            T = T.dot(self.param_manager.Tdof2dof[j])
        T = T.dot(self.param_manager.Tdof2vdof[i_sensor]).dot(self.param_manager.Tvdof2su[i_sensor])

        I = np.eye(3)
        vectors = np.dot(T.R, I)

        return T.position, vectors
    
    def get_all_accelerometer_positions(self):
        """
        Returns all accelerometer positions in the initial position

        Returns 
        --------
        positions: np.ndarray
            All accelerometer positions
        """
        accelerometer_positions = np.zeros((self.n_sensor, 3))
        for i in range(self.n_sensor):
            position, _ = self.get_i_accelerometer_position(i)
            accelerometer_positions[i, :] = position

        return accelerometer_positions

def collect_data(robot):
    """
    Function for collecting acceleration data with poses

    Returns
    --------
    data: Data
        Data includes static and dynamic accelerations data
    poses: np.ndarray
        For all poses for all joints
    """
    ros_robotic_skin_path = rospkg.RosPack().get_path('ros_robotic_skin')
    directory = os.path.join(ros_robotic_skin_path, 'data')

    filename = '_'.join(['static_data', robot])
    filepath = os.path.join(directory, filename + '.pickle')
    with open(filepath, 'rb') as f:
        static = pickle.load(f, encoding='latin1')

    filename = '_'.join(['dynamic_data', robot])
    filepath = os.path.join(directory, filename + '.pickle')
    with open(filepath, 'rb') as f:
        dynamic = pickle.load(f, encoding='latin1')

    Data = namedtuple('Data', 'static dynamic')
    data = Data(static, dynamic)
    poses = np.array([
        [0,     0,    0,    0,    0,    0,    0, ],
        [3.47, -2.37, 1.38, 0.22, 3.13, 1.54, 1.16],
        [-1.10, -2.08, 5.68, 1.41, 4.13, 0.24, 2.70],
        [-0.75, -1.60, 1.56, 4.43, 1.54, 4.59, 6.61],
        [-0.61, -0.54, 3.76, 3.91, 5.05, 0.92, 6.88],
        [-1.39, -0.87, 4.01, 3.75, 5.56, 2.98, 4.88],
        [1.51, -2.47, 3.20, 1.29, 0.24, 4.91, 8.21],
        [0.25, -0.18, 5.13, 5.43, 2.78, 3.86, 6.72],
        [0.76, -1.96, 2.24, 1.54, 4.19, 5.22, 7.46],
        [0.03, -1.09, 2.63, 0.33, 3.87, 0.88, 2.92],
        [0.72, -1.00, 6.09, 2.61, 1.10, 4.13, 3.06],
        [1.69, -2.72, 0.14, 1.08, 2.14, 0.08, 9.13],
        [0.81, -1.89, 3.26, 1.42, 5.64, 0.14, 8.34],
        [-0.90, -3.10, 3.24, 0.16, 4.81, 4.94, 4.35],
        [1.36, -1.89, 2.73, 1.20, 3.08, 3.29, 3.88],
        [-0.36, -2.19, 3.91, 0.04, 2.15, 3.19, 5.18],
        [5.25, -0.55, 0.98, 4.15, 5.65, 3.65, 9.27],
        [2.52, -2.54, 2.07, 0.55, 3.26, 2.31, 4.72],
        [4.63, -0.70, 3.14, 3.41, 3.55, 0.69, 6.10],
        [5.41, -0.90, 5.86, 0.41, 1.69, 1.23, 4.34]
    ])

    if robot == 'panda':
        filepath = os.path.join(directory, 'positions.txt')
        poses = np.loadtxt(filepath)

    return data, poses

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
    measured_data, orientations = collect_data(robot)
    dhparams = load_dhparams(robot)
    estimator = KinematicEstimator(measured_data, orientations, dhparams)
    estimator.optimize()
    Tdof2dof, Tdof2vdof, Tvdof2su = estimator.get_tmat()
    positions = estimator.get_all_accelerometer_positions()
    
    """
    for ind, point in enumerate(positions):
        print(str(ind)+'th SU: [%02.2f, %02.2f, %02.2f]'%(point[0], point[1], point[2]))
    """
