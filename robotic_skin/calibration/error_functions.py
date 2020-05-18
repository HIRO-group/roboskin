import logging
import numpy as np
import robotic_skin.const as C
import pyquaternion as pyqt
from robotic_skin.calibration.utils.io import n2s
from robotic_skin.calibration.utils.quaternion import np_to_pyqt


def estimate_acceleration_analytically(kinemaic_chain, d_joint, i_su, curr_w):
    """
    Estimates the acceleration analytically.

    Arguments
    ---------
    `kinematic_chain`: `robotic_skin.calibration.kinematic_chain.KinematicChain`
        Robot's Kinematic Chain
    `d_joint`: `int`
        dof `d`
    `i`: `int`
        imu `i`
    `curr_w`: `int`
        Angular velocity
    """
    rs_T_su = kinemaic_chain.compute_su_TM(
        i_su=i_su, pose_type='current')

    dof_T_su = kinemaic_chain.compute_su_TM(
        start_joint=d_joint,
        i_su=i_su,
        pose_type='current')

    # Every joint rotates along its own z axis
    w_dofd = np.array([0, 0, curr_w])
    a_dofd = np.cross(w_dofd, np.cross(w_dofd, dof_T_su.position))

    g_rs = np.array([0, 0, 9.81])
    g_su = np.dot(rs_T_su.R.T, g_rs)
    a_centic_su = np.dot(dof_T_su.R.T, a_dofd)

    a_su = a_centic_su + g_su

    return a_su


def estimate_acceleration_numerically(kinematic_chain, d_joint, i_su, curr_w, max_w, joint_angle_func,
                                      apply_normal_mittendorder=False):
    """
    Compute an acceleration value from positions.
    .. math:: `a = \frac{f({\Delta t}) + f({\Delta t) - 2 f(0)}{h^2}`

    This equation came from Taylor Expansion to get the second derivative from f(t).
    .. math:: f(t+{\Delta t}) = f(t) + hf^{\prime}(t) + \frac{h^2}{2}f^{\prime\prime}(t)
    .. math:: f(t-{\Delta t}) = f(t) - hf^{\prime}(t) + \frac{h^2}{2}f^{\prime\prime}(t)

    Add both equations and plug t=0 to get the above equation

    Arguments
    ------------
    `kinematic_chain`: `robotic_skin.calibration.kinematic_chain.KinematicChain`
        Robot's Kinematic Chain
    `d_joint`: `int`
        dof `d`
    `i`: `int`
        imu `i`
    `curr_w`: `int`
        Angular velocity
    apply_normal_mittendorfer: bool
        determines if we resort to the normal method
        mittendorfer uses (which we modified due to some possible missing terms

    Returns
    ---------
    acceleration: np.array
        Acceleration computed from positions
    """  # noqa: W605
    rs_T_su = kinematic_chain.compute_su_TM(
        i_su=i_su, pose_type='current')

    dof_T_su = kinematic_chain.compute_su_TM(
        start_joint=d_joint,
        i_su=i_su,
        pose_type='current')

    # rotation matrix of reference segment to skin unit
    su_R_rs = rs_T_su.R.T

    # Compute Acceleration at RS frame
    # dt should be a small value, recommended to use 1/(1000 * freq)
    dt = 1.0/1000.0

    positions = []
    for t in [dt, -dt, 0]:
        angle = joint_angle_func(curr_w, max_w, t)
        dof_T_dof, rs_T_dof = kinematic_chain.get_current_TMs()
        kinematic_chain.add_a_pose(
            i_joint=d_joint,
            pose=angle,
            dof_T_dof=dof_T_dof,
            rs_T_dof=rs_T_dof)
        T = kinematic_chain._compute_su_TM(i_su, dof_T_dof, rs_T_dof)
        positions.append(T.position)

    # get acceleration and include gravity
    accel_rs = ((positions[0] + positions[1] - 2*positions[2]) / (dt**2))

    gravity = np.array([0, 0, 9.81])
    accel_rs += gravity

    if apply_normal_mittendorder:
        return np.dot(su_R_rs, accel_rs)

    # we need centripetal acceleration here.
    w_dofd = np.array([0, 0, curr_w])
    a_dofd = np.cross(w_dofd, np.cross(w_dofd, dof_T_su.position))

    # Every joint rotates along its own z axis, one joint moves at a time
    # rotate into su frame
    a_centric_su = np.dot(dof_T_su.R.T, a_dofd)
    a_tan_su = np.dot(su_R_rs, accel_rs)
    accel_su = a_centric_su + a_tan_su
    # estimate acceleration of skin unit
    return accel_su


def max_acceleration_joint_angle(curr_w, amplitude, t):
    """
    max acceleration along a joint angle of robot function.
    includes pattern
    """
    # th_pattern = np.sign(t) * max_w / (curr_w) * (1 - np.cos(curr_w*t))
    # th_pattern = np.sign(t) * max_w / (2*np.pi*C.PATTERN_FREQ) * (1 - np.cos(2*np.pi*C.PATTERN_FREQ*t))
    th_pattern = (amplitude / (2*np.pi*C.PATTERN_FREQ)) * (1 - np.cos(2*np.pi*C.PATTERN_FREQ*t))
    # print('-'*20, th_pattern, curr_w, '-'*20)
    return th_pattern


def constant_velocity_joint_angle(curr_w, max_w, t):
    """
    Returns transformation matrix given `t` and current
    angular velocity `curr_w`
    """
    return curr_w*t


class ErrorFunction():
    """
    Error Function class used to evaluate kinematics
    estimation models.
    """
    def __init__(self, data, loss_func):
        """
        Parses the data and gets the loss function.
        """
        self.data = data
        self.loss_func = loss_func

        self.pose_names = list(data.constant.keys())
        self.joint_names = list(data.constant[self.pose_names[0]].keys())
        self.imu_names = list(data.constant[self.pose_names[0]][self.joint_names[0]].keys())
        self.n_dynamic_pose = len(list(data.dynamic.keys()))
        self.n_constant_pose = len(list(data.constant.keys()))
        self.n_static_pose = len(list(data.static.keys()))

        self.n_joint = len(self.joint_names)
        self.n_sensor = self.n_joint

    def __call__(self, kinematic_chain, i_su):
        """
        __call__ is to be used on returning an error value.
        """
        raise NotImplementedError()


class StaticErrorFunction(ErrorFunction):
    """
    Static error is an deviation of the gravity vector for p positions.

    """
    def __init__(self, data, loss_func):
        super().__init__(data, loss_func)

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
        i_joint = kinematic_chain.su_joint_dict[i_su]
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
            accel_rs = np.dot(rs_R_su, accel_su)
            gravities[p, :] = accel_rs
            # Account of Quaternion
            q_su = self.data.static[self.pose_names[p]][self.imu_names[i_su]][:4]
            # d = pyqt.Quaternion.absolute_distance(T.q, np_to_pyqt(q_su))
            d = np.linalg.norm(q_su - T.quaternion)
            # logging.debug(f'Measured: {q_su}, Model: {T.quaternion}')
            error_quaternion[p] = d

        # return np.sum(np.linalg.norm(gravities - np.mean(gravities, 0), axis=1))
        # return np.sum(np.linalg.norm(gravities - gravity, axis=1))
        return self.loss_func(gravities, gravity, axis=1)
        # return self.loss_func(gravities, gravity, axis=1) + np.linalg.norm(error_quaternion)
        # return np.linalg.norm(error_quaternion)


class ConstantRotationErrorFunction(ErrorFunction):
    """
    An error function used when a robotic arm's joints
    are moving at a constant velocity.
    """
    def __init__(self, data, loss_func):
        super().__init__(data, loss_func)

    def __call__(self, kinematic_chain, i_su):
        """
        Arguments
        ------------
        i_su: int
            i_suth sensor
        kinemaic_chain:
            A Kinematic Chain of the robot

        Returns
        --------
        e1: float
            Static Error
        """
        i_joint = kinematic_chain.su_joint_dict[i_su]

        errors = 0.0
        n_data = 0
        for p in range(self.n_constant_pose):
            # for d in range(i+1):
            for d_joint in range(max(0, i_joint-2), i_joint+1):
                data = self.data.constant[self.pose_names[p]][self.joint_names[d_joint]][self.imu_names[i_su]][0]
                # meas_qs = data[:, :4]
                meas_accels = data[:, 4:7]
                joints = data[:, 7:14]
                angular_velocities = data[:, 14]

                # for meas_accel, poses, curr_w in zip(meas_accels, joints, angular_velocities):
                n_eval = 10
                for i in range(n_eval):
                    n_data = data.shape[0]
                    if n_data <= i:
                        break

                    idx = i*int(n_data/n_eval)
                    meas_accel = meas_accels[idx, :]
                    poses = joints[idx, :]
                    curr_w = angular_velocities[idx]

                    # TODO: parse start_joint. Currently, there is a bug
                    kinematic_chain.set_poses(poses, end_joint=i_joint)
                    model_accel = estimate_acceleration_analytically(kinematic_chain, d_joint, i_su, curr_w)

                    logging.debug(f'[Pose{p}, Joint{d_joint}, SU{i_su}@Joint{i_joint}, Data{idx}]\t' +
                                  f'Model: {n2s(model_accel, 4)} SU: {n2s(meas_accel, 4)}')

                    error2 = self.loss_func(model_accel, meas_accel)

                    errors += error2
                    n_data += 1

        return errors/n_data


class MaxAccelerationErrorFunction(ErrorFunction):
    """
    Compute errors between estimated and measured max acceleration for sensor i

    """
    def __init__(self, data, loss_func, apply_normal_mittendorfer=False):
        super().__init__(data, loss_func)
        self.apply_normal_mittendorfer = apply_normal_mittendorfer

    def __call__(self, kinematic_chain, i_su):
        """
        Compute errors between estimated and measured max acceleration for sensor i

        .. math:: `\Sigma_{p=1}^P\Sigma_{d=i-3, i>0}^i {}^{SU_i}|a_{max}^{model} - a_{max}^{measured}|_{i,d,p}^2`

        Arguments
        ------------
        i_su: int
            i_su th sensor
        kinemaic_chain:
            A Kinematic Chain of the robot

        Returns
        --------
        e2: float
            Dynamic Error
        """  # noqa: W605
        i_joint = kinematic_chain.su_joint_dict[i_su]

        e2 = 0.0
        n_data = 0
        for p in range(self.n_dynamic_pose):
            for d_joint in range(max(0, i_joint-2), i_joint+1):
                # max acceleration (x,y,z) of the data
                max_accel_train = self.data.dynamic[self.pose_names[p]][self.joint_names[d_joint]][self.imu_names[i_su]][0][:3]

                curr_w = self.data.dynamic[self.pose_names[p]][self.joint_names[d_joint]][self.imu_names[i_su]][0][5]
                # A is used as amplitude of pose pattern
                A = self.data.dynamic[self.pose_names[p]][self.joint_names[d_joint]][self.imu_names[i_su]][0][4]
                poses = self.data.dynamic[self.pose_names[p]][self.joint_names[d_joint]][self.imu_names[i_su]][0][7:14]

                # kinematic_chain.set_poses(joints)
                kinematic_chain.set_poses(poses, end_joint=i_joint)
                # use mittendorfer's original or modified based on condition
                max_accel_model = estimate_acceleration_numerically(kinematic_chain, d_joint, i_su, curr_w, A, max_acceleration_joint_angle,
                                                                    self.apply_normal_mittendorfer)
                logging.debug(f'[Pose{p}, Joint{d_joint}, SU{i_su}@Joint{i_joint}]\t' +
                              f'Model: {n2s(max_accel_model, 4)} SU: {n2s(max_accel_train, 4)}')
                error = np.sum(np.abs(max_accel_train - max_accel_model)**2)
                e2 += error
                n_data += 1

        return e2/n_data
