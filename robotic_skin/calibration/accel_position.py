#!/usr/bin/env python
"""
Module for Kinematics Estimation
"""
from collections import namedtuple
import nlopt
import numpy as np

import robotic_skin.const as C
from robotic_skin.calibration.utils import TransMat
# need to import set_franka_pose, oscillate_franka

class KinematicEstimator():
    """
    Class for estimating the kinematics of the arm
    and sensor unit positions.
    """
    def __init__(self, data, poses):
        """
        Parameters 
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
        self.n_pose = poses.shape[0]
        self.n_joint = self.n_pose
        self.n_sensor = self.n_pose

        # bounds for DH parameters
        self.bounds = np.array([
            #    th,   d,    a,     al
            [-np.pi, -1.0, 0, -np.pi],
            [np.pi, 1.0, 1.0, np.pi]])
        self.param_manager = ParameterManager(self.n_joint, poses, self.bounds)

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
        for i in range(self.n_joint):
            params, bounds = self.param_manager.get_params_at(i=i)
            n_param = params.shape[0]

            if i == 0:
                assert n_param == 6
            else:
                assert n_param == 10

            Tdofs, Tposes = self.param_manager.get_tmat_until(i)

            # Construct an global optimizer
            opt = nlopt.opt(C.GLOBAL_OPTIMIZER, n_param)
            # The objective function only accepts x and grad arguments. 
            # This is the only way to pass other arguments to opt
            # https://github.com/JuliaOpt/NLopt.jl/issues/27
            opt.set_min_objective(lambda x, grad: self.error_function(x, grad, i, Tdofs, Tposes))

            # Set boundaries
            opt.set_lower_bounds(bounds[0, :])
            opt.set_upper_bounds(bounds[1, :])
            opt.set_stopval(C.GLOBAL_STOP)
            # Need to set a local optimizer for the global optimizer
            local_opt = nlopt.opt(C.LOCAL_OPTIMIZER, n_param)
            local_opt.set_stopval(C.LOCAL_STOP)
            opt.set_local_optimizer(local_opt)

            params = opt.optimize(params)
            # save the optimized parameter to the parameter manager
            self.param_manager.set_params_at(i, params)

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
    
    def get_accelerometer_positions(self):
        """
        Returns all accelerometer positions in the initial position

        Returns 
        --------
        positions: np.ndarray
            All accelerometer positions
        """
        accelerometer_positions = np.zeros((self.n_sensor, 3))
        for i in range(self.n_sensor):
            T = TransMat(np.zeros(4))
            for j in range(i):
                T = self.param_manager.Tdof2dof[j].dot(T)
            T = self.param_manager.Tvdof2su[i].dot(self.param_manager.Tdof2vdof[i].dot(T))

            position = T[:3, 3]
            accelerometer_positions[i, :] = position

        return accelerometer_positions

    def error_function(self, params, i, Tdofs, Tposes, grad=None):
        """
        Computes an error e_T = e_1 + e_2 from current parameters

        Parameters
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
        # Since Tdof2su includes Tranformation Matrix for 2 joints
        # Move them to Tdofs so that len(Tdofs) == len(Tjoints)
        # Refer to Fig. 4
        if grad is not None:
            # do something separate
            pass
        if params.shape[0] == 6:
            # 6 parameters
            #   (2)     (4)
            # su <- vdof <- dof
            Tdofs += [TransMat(params[:4])]
            Tdof2su_i = TransMat(params[4:])
        else:
            # 10 parameters
            #   (2)     (4)    (4)
            # su <- vdof <- dof <- dof
            Tdofs += [TransMat(params[:4]), TransMat(params[4:8])]
            Tdof2su_i = TransMat(params[8:])

        e1 = self.static_error_function(i, Tdof2su_i, Tdofs, Tposes)
        e2 = self.dynamic_error_function(i, Tdof2su_i, Tdofs, Tposes)

        return e1 + e2

    def static_error_function(self, i, Tdof2su_i, Tdofs, Tposes):
        """ 
        Computes static error for ith accelerometer. 
        Static error is an deviation of the gravity vector for p positions. 
        
        This function implements Equation 15 in the paper. 
        .. math:: `e_1 = \Sigma_{p=1}^P |{}^{RS}g_{N,p} - \Sigma_{p=1}^P {}^{RS}g_{N,p}|^2`
        where
        .. math:: `{}^{RS}g_{N,p} = {}^{RS}R_{SU_N}^{mod,p} {}^{SU_N}g_{N,p}`


        Parameters
        ------------
        i: int
            ith sensor
        Tdof2su_i: TransMat
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
        gravities = np.zeros((self.n_pose, 3))

        # loop over P poses
        p = 0
        for p, Tpose in enumerate(Tposes):
            # 1 Pose are consists for n_joint DoF
            T = TransMat(np.zeros(4))   # equals to I Matrix
            for Tdof, Tjoint in zip(Tdofs, Tpose):
                T = Tjoint.dot(Tdof.dot(T))
            # DoF to SU
                T = Tdof2su_i.dot(T)

            Rdof2su = T.R
            accel_su = self.data.static[p, i]
            accel_rs = np.dot(Rdof2su.T, accel_su)
            gravities[p, :] = accel_rs

        return np.sum(np.square(gravities[p, :] - np.mean(gravities, 0)))

    def dynamic_error_function(self, i, Tdof2su_i, Tdofs, Tposes):
        """
        Compute errors between estimated and measured max acceleration for sensor i

        .. math:: `\Sigma_{p=1}^P\Sigma_{d=i-3, i>0}^i {}^{SU_i}|a_{max}^{model} - a_{max}^{measured}|_{i,d,p}^2`
        
        Parameters
        ------------
        i: int
            ith sensor
        Tdof2su_i: TransMat
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

        e2 = 0
        for p, Tpose in enumerate(Tposes):
            for d in range(max(0, i-3), i):
                max_accel_train = self.data.dynamic[p, d, i] 
                max_accel_model = self.estimate_max_acceleration(d, Tdofs, Tpose, Tdof2su_i)
                error = np.square(max_accel_train - max_accel_model)
                e2 += error

        return e2

    def estimate_max_acceleration(self, d, Tdofs, Tjoints, Tdof2su_i):
        """
        Compute an acceleration value from positions.
        .. math:: `a = \frac{f({\Delta t}) + f({\Delta t) - 2 f(0)}{h^2}`

        This equation came from Taylor Expansion.
        .. math:: f(t+{\Delta t}) = f(t) + hf^{\prime}(t) + \frac{h^2}{2}f^{\prime\prime}(t)
        .. math:: f(t-{\Delta t}) = f(t) - hf^{\prime}(t) + \frac{h^2}{2}f^{\prime\prime}(t)

        Add both equations and plug t=0 to get the above equation
        
        Parameters
        ------------
        d: int
            dth excited joint
        Tdof2su_i: TransMat
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
        dt = 1/(1000*C.PATTERN_FREQ)
        pos = lambda dt: self.accelerometer_position(dt, d, Tdofs, Tjoints, Tdof2su_i)
        return (pos(dt) + pos(-dt) - 2*pos(0)) / dt^2

    def accelerometer_position(self, t, d, Tdofs, Tjoints, Tdof2su_i):
        """
        Compute ith accelerometer position excited by joint d in pose p at time t

        At pose p, let o be a joint and x be a sensor unit, then it looks like
                d       i                     dth joint and ith accelerometer
                
        |o-x-o      -x-o      -x-o      -x-
            \-x-o/    \-x-o/    \-x-o/

        1   2    3    4    5    6    7         th joint
        1    2    3    4    5    6    7      th sensor

        Parameters
        ------------
        d: int
            dth excited joint
        Tdof2su_i: TransMat
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
        th_pattern = C.PATTERN_A/(2*np.pi*C.PATTERN_FREQ) * (1 - np.cos(2*np.pi*C.PATTERN_FREQ*t))
        Tpatt = TransMat(th_pattern) # for all joint

        T = TransMat(np.zeros(4))   # equals to I Matrix
        # loop over all joint to get to the ith sensor
        for i_joint in range(self.n_joint):
            # 1. Transform each joint by dh parameter theta Tdofs[i_joint]
            # 2. Then Rotate the axis by theta_pose Tjoints[i_joint] defined by the pose
            T = Tjoints[i_joint].dot(Tdofs[i_joint].dot(T))
            # If at the dth joint, Rotate the joint by theta_pattern
            if i_joint == d:
                T = Tpatt.dot(T)

        # At the end, Transform from the last ith DoF to ith SU
        T = Tdof2su_i.dot(T)

        # Return only the XYZ position of the sensor in Reference Frame
        return T[:3, 3]

class ParameterManager():
    """
    Class for managing DH parameters
    """
    def __init__(self, n_joint, poses, bounds):
        """
        TODO For now, we assume n_sensor is equal to n_joint
        
        Parameters
        -----------
        n_joints: int
            Number of joints
        poses: np.ndarray
            Poses for n_joints
        bounds: np.ndarray
            Bounds for DH parameters
        """
        self.n_joint = n_joint
        # for later: initialize with randomized value within a certain range
        self.poses = poses
        self.bounds = bounds
        self.Tdof2dof = [TransMat() for i in range(n_joint-1)]
        self.Tdof2vdof = [TransMat() for i in range(n_joint)]
        self.Tvdof2su = [TransMat(np.random.rand(2)) for i in range(n_joint)]
        self.Tposes = [[TransMat(np.array(theta)) for theta in pose] for pose in poses]

    def get_params_at(self, i):
        """
        if n_joint is 7 DoF i = 0, 1, ..., 6

        Parameters
        ---------------
        i: int
            ith joint (ith sensor) 
        
        Returns
        --------
        params: np.array
            Next DH parameters to be optimized
        """
        if i == 0:
            params = np.r_[self.Tdof2vdof[i].parameters, self.Tvdof2su[i].parameters]
            bounds = np.hstack([self.bounds[:, :], self.bounds[:, :2]])
        else:
            params = np.r_[self.Tdof2dof[i-1].parameters, 
                           self.Tdof2vdof[i].parameters, 
                           self.Tvdof2su[i].parameters]
            bounds = np.hstack([self.bounds[:, :], self.bounds[:, :], self.bounds[:, :2]])

        return params, bounds

    def get_tmat_until(self, i):
        """
        get transformation matrices when optimizing ith joint (sensor)

        Parameters
        ----------
        i: int
            ith joint (sensor)

        Returns
        --------
        list of TransMat
            Transformation Matrices between DoFs
        list of TransMat
            Transformation Rotation Matrices for all joints
        """
        if i == 0:
            return [TransMat(np.zeros(4))], [self.Tposes[p][:i+1] for p in range(self.poses.shape[0])]
        if i == 1:
            return [TransMat(np.zeros(4))], [self.Tposes[p][:i+1] for p in range(self.poses.shape[0])] 

        return self.Tdof2dof[:i-1], [self.Tposes[p][:i+1] for p in range(self.poses.shape[0])]

    def set_params_at(self, i, params):
        """
        Set DH parameters
        
        Parameters
        ------------
        int: i
            ith joint (sensor)
        parmas: np.array
            DH Parameters
        """
        if i == 0:
            self.Tdof2vdof[i].set_params(params[:4])
            self.Tvdof2su[i].set_params(params[4:])
        else:
            self.Tdof2dof[i-1].set_params(params[:4])
            self.Tdof2vdof[i].set_params(params[4:8])
            self.Tvdof2su[i].set_params(params[8:])

def collect_data():
    """
    Function for collecting acceleration data with poses
    TODO
    Randomly generates for now

    Returns
    --------
    data: Data
        Data includes static and dynamic accelerations data
    poses: np.ndarray
        For all poses for all joints
    """
    Data = namedtuple('Data', 'static dynamic')
    data = Data(np.random.rand(20, 7, 3), np.random.rand(20, 7, 7, 1))
    poses = np.random.rand(20, 7)
    return data, poses

if __name__ == '__main__':
    # Need data
    measured_data, orientations = collect_data()
    estimator = KinematicEstimator(measured_data, orientations)
    estimator.optimize()
    Tdof2dof, Tdof2vdof, Tvdof2su = estimator.get_tmat()
    positions = estimator.get_accelerometer_positions()

    for ind, point in enumerate(positions):
        print(str(ind)+'th SU: [%02.2f, %02.2f, %02.2f]'%(point[0], point[1], point[2]))
