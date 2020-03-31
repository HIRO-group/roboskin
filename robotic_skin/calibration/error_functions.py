import numpy as np
import robotic_skin.const as C
from robotic_skin.calibration.utils import TransMat, get_IMU_pose


def estimate_acceleration_analytically(Tdofs, Tjoints, Tdofi2su, d, i, curr_w):
    # Transformation Matrix from su to rs in rs frame
    rs_T_su = TransMat(np.zeros(4))
    # Transformation Matrix from the last DoFi to the excited DoFd
    dofd_T_dofi = TransMat(np.zeros(4))

    for j in range(d+1):
        # print(j)
        rs_T_su = rs_T_su.dot(Tdofs[j]).dot(Tjoints[j])

    for j in range(d+1, i+1):
        # print(j, d, i)
        rs_T_su = rs_T_su.dot(Tdofs[j]).dot(Tjoints[j])
        dofd_T_dofi = dofd_T_dofi.dot(Tdofs[j]).dot(Tjoints[j])

    rs_T_su = rs_T_su.dot(Tdofi2su)
    dof_T_su = dofd_T_dofi.dot(Tdofi2su)

    dofd_r_su = dof_T_su.position
    # Every joint rotates along its own z axis
    w_dofd = np.array([0, 0, curr_w])
    a_dofd = np.cross(w_dofd, np.cross(w_dofd, dofd_r_su))

    g_rs = np.array([0, 0, 9.81])

    a_su = np.dot(dof_T_su.R.T, a_dofd) + np.dot(rs_T_su.R.T, g_rs)

    return a_su


def estimate_acceleration_numerically(Tdofs, Tjoints, Tdof2su, d, curr_w, max_w, joint_angle_func):
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
    pos = lambda dt: accelerometer_position(dt, Tdofs, Tjoints, Tdof2su, d, curr_w, max_w, joint_angle_func)  # noqa: E731
    gravity = np.array([0, 0, 9.81])

    accel_rs = (pos(dt) + pos(-dt) - 2*pos(0)) / (dt**2) + gravity
    accel_su = np.dot(Rrs2su, accel_rs)

    return accel_su


def accelerometer_position(t, Tdofs, Tjoints, Tdof2su, d, curr_w, max_w, joint_angle_func):
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


class ErrorFunction():
    def __init__(self, data, loss_func):
        self.data = data
        self.loss_func = loss_func

        self.pose_names = list(data.constant.keys())
        self.joint_names = list(data.constant[self.pose_names[0]].keys())
        self.imu_names = list(data.constant[self.pose_names[0]][self.joint_names[0]].keys())
        self.n_pose = len(self.pose_names)
        self.n_joint = len(self.joint_names)
        self.n_sensor = self.n_joint

    def __call__(self, i, Tdofs, Tdof2su):
        raise NotImplementedError()


class StaticErrorFunction(ErrorFunction):
    def __init__(self, data, loss_func):
        super().__init__(data, loss_func)

    def __call__(self, i, Tdofs, Tdof2su):
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

        # return np.sum(np.linalg.norm(gravities - np.mean(gravities, 0), axis=1))
        # return np.sum(np.linalg.norm(gravities - gravity, axis=1))
        # return np.mean(np.linalg.norm(gravities - gravity, axis=1))
        return self.loss_func(gravities - gravity)


class ConstantRotationErrorFunction(ErrorFunction):
    def __init__(self, data, loss_func):
        super().__init__(data, loss_func)

    def __call__(self, i, Tdofs, Tdof2su):
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
        errors = 0.0
        n_data = 0
        for p in range(self.n_pose):
            for d in range(max(0, i-2), i+1):
            # for d in range(i+1):
                data = self.data.constant[self.pose_names[p]][self.joint_names[d]][self.imu_names[i]][0]
                meas_qs = data[:, :4]
                meas_accels = data[:, 4:7]
                joints = data[:, 7:14]
                angular_velocities = data[:, 14]

                for meas_q, meas_accel, joint, curr_w in zip(meas_qs, meas_accels, joints, angular_velocities):
                    # Orientation Error
                    """
                    model_q = get_IMU_pose(Tdofs, Tdof2su, joint[:i+1])
                    meas_q = tfquat_to_pyquat(meas_q)
                    error1 = pyqt.Quaternion.absolute_distance(model_q, meas_q)
                    """

                    # Acceleration Error
                    Tjoints = [TransMat(joint) for joint in joint[:i+1]]
                    # model_accel = self.estimate_acceleration_numerically(
                    # Tdofs, Tjoints, Tdof2su, d, curr_w, None, constant_velocity_joint_angle)
                    model_accel = estimate_acceleration_analytically(Tdofs, Tjoints, Tdof2su, d, i, curr_w)
                    # error2 = np.sum(np.abs(model_accel - meas_accel))
                    error2 = np.sum(np.linalg.norm(model_accel - meas_accel))
                    # print(i, d, joint[d], curr_w, n2s(model_accel), n2s(meas_accel))
                    # print(n2s(joint))

                    # errors += error1 + error2
                    errors += error2
                    n_data += 1

        return errors/n_data


class MaxAccelerationErrorFunction(ErrorFunction):
    def __init__(self, data, loss_func):
        super().__init__(data, loss_func)

    def __call__(self, i, Tdofs, Tdof2su):
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
                # max_accel_model = self.estimate_acceleration_numerically(
                # Tdofs, Tjoints, Tdof2su, d, curr_w, max_w, max_acceleration_joint_angle)
                max_accel_model = self.estimate_acceleration_analytically(Tdofs, Tjoints, Tdof2su, d, i, curr_w)
                # if p == 0:
                #     print('[Dynamic Max Accel, %ith Joint]'%(d), n2s(max_accel_train), n2s(max_accel_model), curr_w, max_w)
                error = np.sum(np.abs(max_accel_train - max_accel_model))
                e2 += error
                n_data += 1

        return e2/n_data


class ErrorFunctions():
    def __init__(self):
        pass

    def __call__(self):
        raise NotImplementedError()
