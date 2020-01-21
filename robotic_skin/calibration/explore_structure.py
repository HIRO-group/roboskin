#!/usr/bin/env python
import rospy
import numpy as np

import robotic_skin.const as C


def explore_structure_dependency():
    """
    Main function for Algorithm 1 (Structural Exploration).
    Explores how segments and limbs are connected.
    """
    SUs, DoFs = detect_SU_and_DoF_nums()
    static_accels = collect_initial_accels(SUs, DoFs, C.EXPLORE_SAMPLING_LENGTH)
    accels = collect_accelerometer_values(SUs, DoFs, C.EXPLORE_SAMPLING_LENGTH)
    mean_accels = mean_accelerometer_values(accels)
    A = create_activity_matrix(static_accels, mean_accels)
    A_merged = merge_activity_matrix(A)
    A_sorted = sort_merged_activity_matrix(A_merged)
    sensor_alloc, kinematics = identify_kinematic_structure(A_sorted)

    return sensor_alloc, kinematics

def detect_SU_and_DoF_nums():
    """
    Detects how many Sensor Units (SU) and Degree of Freedoms (DoF) there are
    according to the connected sensor values and joints.
    May need to specify by us.
    """
    raise NotImplementedError()

def collect_initial_accels(SUs, DoFs, T, get_accel):
    """
    Parameters
    ----------
    SUs: int
        Number of sensor units 
    DoFs: int
        Number of Degree of Freedoms
    T: int
        Number of sampling length 
        
    Returns 
    ----------
    static_accels: np.ndarray
        Accelerometer values when static
    """
    # TODO
    static_accels = get_accel()
    return static_accels

def collect_accelerometer_values(SUs, DoFs, T):
    """
    Parameters
    ----------
    SUs: int
        Number of sensor units 
    DoFs: int
        Number of Degree of Freedoms
    T: int
        Number of sampling length 
        
    Returns 
    ----------
    accels: np.ndarray
        n acceleromter sensor values excited by n+1 joints for T time. 
    """
    accels = np.zeros((SUs, DoFs, T))
    for d in range(DoFs):
        accel = excite_joint(d)
        # TODO
        # I believe this does not work. Need tweek.
        accels[:, d, :] = accel

    return accels

def excite_joint(n_joint, angular_vel=C.EXPLORE_ANGULAR_VEL):
    # TODO
    # Need to find a way to command robotic arm's angular velocity
    raise NotImplementedError()

def mean_accelerometer_values(raw_accels):
    """
    Compute the average acceleromter values over time for each sensor excited by each joint.
    Turn (n x n+1 x T) matrix into (n x n+1)

    Parameters
    ----------
    raw_accels: np.ndarray
        n acceleromter sensor values excited by n+1 joints for T time. 
        
    Returns 
    ----------
    mea_accels: np.ndarray
        Mean n acceleromter sensor values excited by n+1 joints
    """
    SUs = raw_accels.shape[0]
    DoFs = raw_accels.shape[1]
    mean_accels = np.zeros((SUs, DoFs))
    for SU in range(SUs):
        for DoF in range(DoFs):
           mean_accels = np.mean(raw_accels[SU, DoF, :])

    return mean_ccels 

def create_activity_matrix(static_accels, measured_accels):
    """
    Generates an activity matrix from accelerometer values
    
    Parameters
    ----------
    static_accels: np.array
        Accelerometer sensor values measured in the initial static position
        (n x 1) 
    measured_accels: np.ndarray
        Accelerometer sensor values (mean value) 
        (n x n+1)

    Returns
    ----------
    A: np.ndarray
        Activity Matrix
    """
    A = np.zeros_like(static_accels)

    DoFs = measured_accels[1]
    for d in range(DoFs):
        A[:, d] = (static_accels != measured_accels[:,d])

    return A

def merge_activity_matrix(A):
    """
    Merges similar rows and columns.
    
    Parameters
    ----------
    A: np.ndarray
        Activity Matrix (n by n+1 array) 
        where n is a number of acclerometers

    Returns
    ----------
    A: np.ndarray
        Merged Activity Matrix
    """
    raise NotImplementedError()

def sort_merged_activity_matrix(A):
    """
    Sorts the merged activity matrix to a lower triangular form. 

    Parameters
    ----------
    A: np.ndarray
        Merged Activity Matrix (n by n+1 array) 
        where n is a number of acclerometers
    
    Returns 
    ----------
    A: np.ndarray
        Sorted merged Activity Matrix (n by n+1 array) 
        where n is a number of acclerometers
    """
    raise NotImplementedError()

def identify_kinematic_structure(A):
    """
    Identify the kinematic structure the given sorted merged activity matrix.
    Kinematic structure provides information about how segments and limbs are connected.
    See Fig. 5 of "Open-loop Self-calibration of Articulated Robots with Artificial Skins"

    Parameters
    ----------
    A: np.ndarray
        Sorted Merged Activity Matrix (n by n+1 array) 
        where n is a number of acclerometers
    
    Returns 
    ----------
    sensor_alloc: dict
        Stores which segment each sensor is allocated to.
        Key is the Sensor Unit ID (SU) and value is Segment ID.
    kinematic: dict
        Stores how each segment is connected to the joints
        Key is the Segment ID and values are Joint ID. 
    """

    raise NotImplementedError()