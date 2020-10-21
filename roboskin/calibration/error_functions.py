import numpy as np
import pyquaternion as pyqt
import roboskin.const as C
from roboskin.calibration.utils.quaternion import np_to_pyqt
from roboskin.calibration.utils.rotational_acceleration import estimate_acceleration
# from roboskin.calibration.utils.io import n2s
# import logging


def max_angle_func(t, i_joint, delta_t=0.0, **kwargs):
    """
    Computes current joint angle at time t
    joint is rotated in a sinusoidal motion during MaxAcceleration Data Collection.

    Parameters
    ------------
    `t`: `int`
        Current time t
    `i_joint`: `int`
    `delta_t`: `float`
    """
    # return joint_angle + t *joint_velocity
    return (C.MAX_ANGULAR_VELOCITY[i_joint] / (2*np.pi*C.PATTERN_FREQ[i_joint])) *\
           (1 - np.cos(2*np.pi*C.PATTERN_FREQ[i_joint] * (t - delta_t)))


class ErrorFunction():
    """
    Error Function class used to evaluate kinematics
    estimation models.
    """
    def __init__(self, loss):
        """
        Parses the data and gets the loss function.
        """
        self.initialized = False
        self.loss = loss

    def initialize(self, data):
        self.initialized = True
        self.data = data
        self.pose_names = list(data.dynamic.keys())
        self.joint_names = list(data.dynamic[self.pose_names[0]].keys())
        self.imu_names = list(data.dynamic[self.pose_names[0]][self.joint_names[0]].keys())
        self.n_dynamic_pose = len(list(data.dynamic.keys()))
        self.n_static_pose = len(list(data.static.keys()))

        self.n_joint = len(self.joint_names)
        self.n_sensor = self.n_joint

    def __call__(self, kinematic_chain, inert_su):
        """
        __call__ is to be used on returning an error value.
        """
        if not self.initialized:
            raise ValueError('Not Initialized')
        raise NotImplementedError()


class StaticErrorFunction(ErrorFunction):
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
        kinematic_chain:
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

        gravities = np.zeros((self.n_static_pose, 3))
        gravity = np.array([[0, 0, 9.8], ] * self.n_static_pose, dtype=float)
        error_quaternion = np.zeros(self.n_static_pose)

        for p in range(self.n_static_pose):
            poses = self.data.static[self.pose_names[p]][self.imu_names[i_su]][7:14]
            kinematic_chain.set_poses(poses)
            T = kinematic_chain.compute_su_TM(i_su, pose_type='current')
            # Account for Gravity
            rs_R_su = T.R
            accel_su = self.data.static[self.pose_names[p]][self.imu_names[i_su]][4:7]
            if np.isnan(accel_su).any():
                continue
            # logging.debug(accel_su)
            accel_rs = np.dot(rs_R_su, accel_su)
            gravities[p, :] = accel_rs
            # Account of Quaternion
            q_su = self.data.static[self.pose_names[p]][self.imu_names[i_su]][:4]

            d = pyqt.Quaternion.absolute_distance(T.q, np_to_pyqt(q_su))
            d = np.linalg.norm(q_su - T.quaternion)
            error_quaternion[p] = d

        return self.loss(gravities, gravity, axis=1)


class MaxAccelerationErrorFunction(ErrorFunction):
    """
    Compute errors between estimated and measured max acceleration for sensor i

    """
    def __init__(self, loss, method='our'):
        super().__init__(loss)
        self.method = method

    def initialize(self, data):
        super().initialize(data)
        self.pose_names = list(data.dynamic.keys())
        self.joint_names = list(data.dynamic[self.pose_names[0]].keys())
        self.imu_names = list(data.dynamic[self.pose_names[0]][self.joint_names[0]].keys())

        if 'mittendorfer' in self.method:
            self.should_use_one_point = True
            self.use_max_accel_point()
        else:
            self.should_use_one_point = False

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
        # Will be add a multiprocessing feature.

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
                joint_angular_velocities = data[:, 13]
                n_eval = 1 if self.should_use_one_point else 4
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
                    estimate_A = estimate_acceleration(
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
                    error = np.sum(np.abs(measured_A - estimate_A)**2)
                    e2 += error
                    n_data += 1

        return e2/n_data

    def use_max_accel_point(self):
        """
        conditions for update of best idx:
            - the norm is greater than the current highest one.
            - the time of this data lies within `time_range`
            - the joint acceleration is also greater than the current highest one.

        explanation:
        we use the information from both the norms of the SU acceleration
        and joint acceleration values. Since alpha x r,
        where alpha is joint acceleration is dominant
        in the calculation of SU acceleration, using both sources of information is
        more reliable and robust than just using one.
        """
        time_range = (0.04, 0.16)
        # filter code.
        for pose_name in self.pose_names:
            for joint_name in self.joint_names:
                for imu_name in self.imu_names:
                    imu_data = self.data.dynamic[pose_name][joint_name][imu_name]

                    imu_accs = imu_data[:, :3]
                    acceleration_norms = np.linalg.norm(imu_accs, axis=1)
                    joint_accs = imu_data[:, 11]

                    # max imu acceleration
                    imu_acc_max = 0
                    # max individual joint acceleration
                    joint_acc_max = 0
                    # idx of the max acceleration.
                    best_idx = 0

                    for idx, (acceleration_norm, joint_acc) in enumerate(zip(acceleration_norms, joint_accs)):
                        cur_time = imu_data[idx, 10]
                        if acceleration_norm > imu_acc_max and cur_time < time_range[1] \
                           and cur_time > time_range[0] and joint_acc > joint_acc_max:
                            best_idx = idx
                            imu_acc_max = acceleration_norm
                            joint_acc_max = joint_acc

                    max_point = self.data.dynamic[pose_name][joint_name][imu_name][best_idx]
                    self.data.dynamic[pose_name][joint_name][imu_name] = np.array([max_point])


class CombinedErrorFunction(ErrorFunction):
    """
    combined error function that allows for
    the error based on the cumulative sum of error
    functions.
    """
    def __init__(self, **kwargs):
        self.error_funcs = []
        for k, v in kwargs.items():
            if not isinstance(v, ErrorFunction):
                raise ValueError('Only ErrorFunction class is allowed')
            setattr(self, k, v)
            self.error_funcs.append(v)

    def initialize(self, data):
        for error_function in self.error_funcs:
            error_function.initialize(data)

    def __call__(self, kinematic_chain, i_su):
        e = 0.0
        for error_function in self.error_funcs:
            e += error_function(kinematic_chain, i_su)
        return e
