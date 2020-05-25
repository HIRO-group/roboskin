# import logging
import numpy as np
import robotic_skin.const as C
import pyquaternion as pyqt
from robotic_skin.calibration.utils.quaternion import np_to_pyqt


def current_su_position(kinematic_chain, curr_w, max_w, i_su, d_joint, t):
    """
    Returns position of the current skin unit

    Arguments
    ---------
    `kinematic_chain`: `robotic_skin.calibration.kinematic_chain.KinematicChain`
        Robot's Kinematic Chain
    `curr_w`: `int`
        Angular velocity
    'max_w': 'int'
        Maximum angular velocity
    `i`: `int`
        imu `i`
    'd_joint': 'int'
        dof 'd'
    """
    angle = (max_w / (2*np.pi*C.PATTERN_FREQ)) * (1 - np.cos(2*np.pi*C.PATTERN_FREQ*t))
    dof_T_dof, rs_T_dof = kinematic_chain.get_current_TMs()
    kinematic_chain.add_a_pose(
        i_joint=d_joint,
        pose=angle,
        dof_T_dof=dof_T_dof,
        rs_T_dof=rs_T_dof)
    T = kinematic_chain._compute_su_TM(i_su, dof_T_dof, rs_T_dof)
    return T.position


def estimate_acceleration(kinematic_chain, d_joint, i_su, curr_w, max_w=0,
                          apply_normal_mittendorfer=False, analytical=True):
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
    'max_w': 'int'
        Maximum angular velocity
    apply_normal_mittendorfer: bool
        determines if we resort to the normal method
        mittendorfer uses (which we modified due to some possible missing terms)
    analytical: bool
        determines if we are returning the analytical or numerical
        estimation of acceleration
    """
    rs_T_su = kinematic_chain.compute_su_TM(
        i_su=i_su, pose_type='current')

    dof_T_su = kinematic_chain.compute_su_TM(
        start_joint=d_joint,
        i_su=i_su,
        pose_type='current')

    # Every joint rotates along its own z axis
    w_dofd = np.array([0, 0, curr_w])
    a_dofd = np.cross(w_dofd, np.cross(w_dofd, dof_T_su.position))

    a_centric_su = np.dot(dof_T_su.R.T, a_dofd)

    # Gravity vector
    gravity = np.array([0, 0, 9.81])

    # rotation matrix of reference segment to skin unit
    su_R_rs = rs_T_su.R.T

    # If the analytical boolean is true
    # Calculate analytical acceleration estimation
    if analytical:
        # Gravity vector of skin unit
        g_su = np.dot(su_R_rs, gravity)

        # Acceleration of skin unit
        a_su = a_centric_su + g_su

        return a_su

    # The following will run if analytical boolean is false

    # Compute Acceleration at RS frame
    # dt should be small value, recommended to use 1/(1000 * freq)
    dt = 1.0 / 1000.0

    positions = []
    for t in [dt, -dt, 0]:
        curr_position = current_su_position(kinematic_chain, curr_w, max_w, i_su, d_joint, t)
        positions.append(curr_position)

    # get acceleration and include gravity
    a_rs = ((positions[0] + positions[1] - 2*positions[2]) / (dt**2))

    a_rs += gravity

    # If necessary, we can change a_rs and a_su  for non-analytical
    # back to accel_rs and accel_su
    a_tan_su = np.dot(su_R_rs, a_rs)

    if apply_normal_mittendorfer:
        return a_tan_su

    # Every joint rotates along its own z axis, one joint moves at a time
    # rotate into su frame
    a_su = a_centric_su + a_tan_su
    # estimate acceleration of skin unit
    return a_su


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
            d = pyqt.Quaternion.absolute_distance(T.q, np_to_pyqt(q_su))
            d = np.linalg.norm(q_su - T.quaternion)
            # logging.debug(f'Measured: {q_su}, Model: {T.quaternion}')
            error_quaternion[p] = d

        return self.loss(gravities, gravity, axis=1)


class ConstantRotationErrorFunction(ErrorFunction):
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
        i_su: int
            i_suth sensor
        kinemaic_chain:
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
                    model_accel = estimate_acceleration(kinematic_chain=kinematic_chain,
                                                        d_joint=d_joint,
                                                        i_su=i_su, curr_w=curr_w)

                    # logging.debug(f'[Pose{p}, Joint{d_joint}, SU{i_su}@Joint{i_joint}, Data{idx}]\t' +
                    #               f'Model: {n2s(model_accel, 4)} SU: {n2s(meas_accel, 4)}')

                    error2 = self.loss(model_accel, meas_accel)

                    errors += error2
                    n_error += 1

        return errors/n_error


class MaxAccelerationErrorFunction(ErrorFunction):
    """
    Compute errors between estimated and measured max acceleration for sensor i

    """
    def __init__(self, loss, apply_normal_mittendorfer=False):
        super().__init__(loss)
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
        if not self.initialized:
            raise ValueError('Not Initialized')

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
                max_accel_model = estimate_acceleration(kinematic_chain=kinematic_chain,
                                                        d_joint=d_joint,
                                                        i_su=i_su, curr_w=curr_w,
                                                        max_w=A,
                                                        apply_normal_mittendorfer=self.apply_normal_mittendorfer,
                                                        analytical=False)

                logging.debug(f'[Pose{p}, Joint{d_joint}, SU{i_su}@Joint{i_joint}]\t' +
                              f'Model: {n2s(max_accel_model, 4)} SU: {n2s(max_accel_train, 4)}')
                error = np.sum(np.abs(max_accel_train - max_accel_model)**2)
                e2 += error
                n_data += 1

        return e2/n_data


class CombinedErrorFunction(ErrorFunction):
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
