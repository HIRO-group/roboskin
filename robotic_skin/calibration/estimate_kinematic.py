#!/usr/bin/env python
import rospy
import nlopt
import numpy as np
from collections import namedtuple

import robotic_skin.const as C
from robotic_skin.calibration.explore_structure import explore_structure_dependency


def estimate_kinematic():
    """
    """
    sensor_alloc, kinematics = explore_structure_dependency()
    data = generate_traning_data()
    params = optimize_parameters(data)

    return params

def generate_traning_data(n_pose=C.N_POSE, n_joint=C.N_JOINT, T=C.EXPLORE_SAMPLING_LENGTH):
    """
    Generate training data by oscillating a joint.
    POS_NUM numbers of initial poses will be set.
    At each pose, all joints are actuated one by one.
    For the order, please refer to Fig.4 in the paper.

    Parameters
    ----------
    n_pose: int
        Number of poses
    n_joint: int 
        Number of joints
        
    Returns 
    ----------
    data: np.ndarray
        Training data consists of static and dynamic accelerometer values.
    """
    poses = 2*np.pi*np.random.rand(n_pose, n_joint) - np.pi
    static_accels = np.zeros()
    dynamic_accels = np.zeros()

    for p, pose in enumerate(poses):
        # TODO
        set_pose(pose)
        static_accel = measure_accels()

        dynamic_accels = np.array((n_joint, T))
        for t in range(T):
            oscillate()
            accels = measure_accels()
            dynamic_accel[:, t] =  accels

        static_accels[p, :] = static_accel
        dynamic_accels[p, :] = dynamic_accel 

    Data = namedtuple('Data', "static dynamic")
    return Data(static=static_accels, dynamic=dynamic_accels)

def optimize_parameters(data):

    for sensor in sensors:
        # Construct an optimizer (LDS-based MLSL)
        opt = nlopt.opt(nlopt.NLOPT_G_MLSL_LDS, n_param)
        # This is the only way to pass data to the error function
        # https://github.com/JuliaOpt/NLopt.jl/issues/27
        opt.set_min_objective(lambda x, grad: error_function(x, grad, data))

        # Set boundaries
        lb = -np.pi * np.ones(n_joint)
        ub =  np.pi * np.ones(n_joint)
        opt.set_lower_bounds(lb)
        opt.set_upper_bounds(ub)
        opt.set_stopval(C.GLOBAL_STOP)

        local_opt = nlopt.opt(nlop.NLOPT_LN_NELDERMEAD, n_param)
        local_opt.set_stopval(C.LOCAL_STOP)
        opt.set_local_optimizer(local_opt)

        params = opt.optimize(init_params)

    raise NotImplementedError()

def error_function(params, grad, data):
    """
    computes an error e_T = e_1 + e_2 from current parameters

    Parameters
    ----------
    params: np.ndarray
        Current estimated parameters
    grad: np.ndarray
        Gradient

    Returns 
    ----------
    """
    e1 = static_error_function(params, data, nth_accel, poses)
    e2 = dynamic_error_function(params, data, nth_accel, poses)

    return e1 + e2

def static_error_function(params, data, nth_accel, poses):
    """
    Computes static error for nth accelerometer. 
    Static error is an deviation of the gravity vector for p positions. 
    """
    gravities = data.static
    # Take an average over time T
    # shape = (P x 1 x 1) = (P, )
    gravity_su = np.mean(gravities[:, nth_accel, :], 2)
    gravity_rs = np.dot(rotation_matrix(params, nth_accel, poses), gravity_su)

    return np.sum(np.square(gravity - np.mean(gravity)))

def dynamic_error_function():
    raise NotImplementedError

def rotation_matrix(params, nth_accel, poses):
    raise NotImplementedError