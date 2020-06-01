import logging
import numpy as np
import torch
import robotic_skin.const as C
import pyquaternion as pyqt
from robotic_skin.calibration.error_functions import ErrorFunction
from robotic_skin.calibration.utils.io import n2s, t2s
from robotic_skin.calibration.utils.quaternion import np_to_pyqt
from robotic_skin.calibration.utils.rotational_acceleration import estimate_acceleration
from robotic_skin.calibration.utils.rotational_acceleration_torch import estimate_acceleration_torch


def max_angle_func(t: int):
    """
    Computes current joint angle at time t
    joint is rotated in a sinusoidal motion during MaxAcceleration Data Collection.

    Parameters
    ------------
    `t`: `int`
        Current time t
    """
    return (C.MAX_ANGULAR_VELOCITY / (2*np.pi*C.PATTERN_FREQ)) * (1 - np.cos(2*np.pi*C.PATTERN_FREQ * t))


class StaticErrorFunctionTorch(ErrorFunction):
    """
    Static error is an deviation of the gravity vector for p positions.

    """
    def __init__(self, loss):
        super().__init__(loss)

    def __call__(self, kinematic_chain, i_su):
        """
        Computes static error for ith accelerometer.
        Static error is an deviation of the gravity vector for p positions.

        This function implements Equation 15 in the paper.
        .. math:: `e_1 = \Sigma_{p=1}^P |{}^{RS}g_{N,p} - \Sigma_{p=1}^P {}^{RS}g_{N,p}|^2`
        where
        .. math:: `{}^{RS}g_{N,p} = {}^{RS}R_{SU_N}^{mod,p} {}^{SU_N}g_{N,p}`


        Arguments
        ------------
        kinemaic_chain:
            A Kinematic Chain of the robot
        i_su: int
            i_su th sensor

        Returns
        --------
        e1: float
            Static Error

        """  # noqa: W605
        if not self.initialized:
            raise ValueError('Not Initialized')

        gravities = torch.zeros((self.n_static_pose, 3)).double().cuda()

        gravity = torch.tensor([[0, 0, 9.8], ] * self.n_static_pose).double().cuda()

        error_quaternion = torch.zeros(self.n_static_pose).double().cuda()


        for p in range(self.n_static_pose):
            poses = self.data.static[self.pose_names[p]][self.imu_names[i_su]][7:14]
            kinematic_chain.set_poses(poses)
            T = kinematic_chain.compute_su_TM(i_su, pose_type='current')
            # Account for Gravity
            rs_R_su = T.R
            accel_su = self.data.static[self.pose_names[p]][self.imu_names[i_su]][4:7]

            accel_su = torch.Tensor(accel_su).double().cuda()

            # rotate accel_su into rs frame.
            accel_rs = torch.mm(rs_R_su, accel_su.view(3, 1)).view(-1)

            gravities[p, :] = accel_rs
            # Account of Quaternion
            q_su = self.data.static[self.pose_names[p]][self.imu_names[i_su]][:4]
            d = pyqt.Quaternion.absolute_distance(T.q, np_to_pyqt(q_su))
            d = np.linalg.norm(q_su - T.quaternion)
            # logging.debug(f'Measured: {q_su}, Model: {T.quaternion}')
            error_quaternion[p] = d

        return self.loss(gravities, gravity)


class ConstantRotationErrorFunctionTorch(ErrorFunction):
    """
    An error function used when a robotic arm's joints
    are moving at a constant velocity.
    """
    def __init__(self, loss):
        super().__init__(loss)

    def __call__(self, kinematic_chain, i_su):
        """
        Arguments
        ------------
        i_su: `int`
            i_su-th sensor
        kinematic_chain:
            A Kinematic Chain of the robot

        Returns
        --------
        e1: float
            Static Error
        """
        if not self.initialized:
            raise ValueError('Not Initialized')

        i_joint = kinematic_chain.su_joint_dict[i_su]

        errors = 0.0
        n_error = 0
        for p in range(self.n_constant_pose):
            # for d in range(i+1):
            for d_joint in range(max(0, i_joint-2), i_joint+1):
                data = self.data.constant[self.pose_names[p]][self.joint_names[d_joint]][self.imu_names[i_su]][0]
                # meas_qs = data[:, :4]
                meas_accels = data[:, 4:7]
                joints = data[:, 7:14]
                angular_velocities = data[:, 14]

                n_eval = 10
                for i in range(n_eval):
                    n_data = data.shape[0]
                    if n_data <= i:
                        break

                    idx = i*int(n_data/n_eval)
                    meas_accel = meas_accels[idx, :]
                    poses = joints[idx, :]
                    angular_velocity = angular_velocities[idx]
                    meas_accel_torch = torch.Tensor(meas_accel).double().cuda()
                    angular_velocity_torch = torch.Tensor(angular_velocity).double().cuda()
                    # TODO: parse start_joint. Currently, there is a bug
                    kinematic_chain.set_poses(poses, end_joint=i_joint)
                    model_accel_torch = estimate_acceleration_torch(kinematic_chain=kinematic_chain,
                                                        i_rotate_joint=d_joint,
                                                        i_su=i_su,
                                                        joint_angular_velocity=angular_velocity_torch)

                    # logging.debug(f'[Pose{p}, Joint{d_joint}, SU{i_su}@Joint{i_joint}, Data{idx}]\t' +
                    #               f'Model: {n2s(model_accel, 4)} SU: {n2s(meas_accel, 4)}')

                    error2 = self.loss(model_accel_torch, meas_accel_torch)

                    errors += error2
                    n_error += 1

        return errors/n_error


class MaxAccelerationErrorFunctionTorch(ErrorFunction):
    """
    Compute errors between estimated and measured max acceleration for sensor i

    """
    def __init__(self, loss, method='normal_mittendorfer'):
        super().__init__(loss)
        self.method = method

    def __call__(self, kinematic_chain, i_su):
        """
        Compute errors between estimated and measured max acceleration for sensor i

        .. math:: `\Sigma_{p=1}^P\Sigma_{d=i-3, i>0}^i {}^{SU_i}|a_{max}^{model} - a_{max}^{measured}|_{i,d,p}^2`

        Arguments
        ------------
        i_su: int
            i_su th sensor
        kinematic_chain:
            A Kinematic Chain of the robot

        Returns
        --------
        e2: float
            Dynamic Error
        """  # noqa: W605
        if not self.initialized:
            raise ValueError('Not Initialized')

        i_joint = kinematic_chain.su_joint_dict[i_su]

        e2 = 0.0
        n_data = 0
        for i_pose in range(self.n_dynamic_pose):
            for rotate_joint in range(max(0, i_joint-2), i_joint+1):
                # max acceleration (x,y,z) of the data
                su = self.imu_names[i_su]
                pose = self.pose_names[i_pose]
                joint = self.joint_names[rotate_joint]

                data = self.data.dynamic[pose][joint][su]
                measured_As = data[:, :3]
                joints = data[:, 3:10]
                times = data[:, 10]
                joint_angular_accelerations = data[:, 11]
                # max_angular_velocity = data[0, 12]
                joint_angular_velocities = data[:, 13]

                n_eval = 4
                for i_eval in range(n_eval):
                    n_data = data.shape[0]
                    if n_data <= i_eval:
                        break

                    idx = i_eval * int(n_data/n_eval)
                    measured_A = measured_As[idx, :]
                    poses = joints[idx, :]
                    time = times[idx]
                    joint_angular_acceleration = joint_angular_accelerations[idx]
                    joint_angular_velocity = joint_angular_velocities[idx]


                    # kinematic_chain.set_poses(joints)
                    kinematic_chain.set_poses(poses, end_joint=i_joint)
                    # use mittendorfer's original or modified based on condition
                    estimate_A_tensor = estimate_acceleration_torch(
                        kinematic_chain=kinematic_chain,
                        i_rotate_joint=rotate_joint,
                        i_su=i_su,
                        joint_angular_velocity=joint_angular_velocity,
                        joint_angular_acceleration=joint_angular_acceleration,
                        current_time=time,
                        angle_func=max_angle_func,
                        method=self.method)

                    # logging.debug(f'[{pose}, {joint}, {su}@Joint{i_joint}]\t' +
                    #               f'Model: {n2s(estimate_A, 4)} SU: {n2s(measured_A, 4)}')
                    measured_A_tensor = torch.tensor(measured_A).double().cuda()
                        # print(max_accel_model.detach().numpy(), max_accel_train)
                    error = torch.sum(torch.abs(measured_A_tensor - estimate_A_tensor)**2)


                    e2 += error
                    n_data += 1

        return e2/n_data
