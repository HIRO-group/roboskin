# import logging
import numpy as np
import robotic_skin.const as C
import pyquaternion as pyqt
import matplotlib.pyplot as plt
from robotic_skin.calibration.utils.quaternion import np_to_pyqt
# from robotic_skin.calibration.utils import n2s


def estimate_acceleration(kinematic_chain, i_rotate_joint, i_su,
                          joint_angular_velocity, joint_angular_acceleration=0,
                          max_angular_velocity=0, current_time=0, method='analytical'):
    r"""
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
    `i_rotate_joint`: `int`
        dof `d`
    `i_su`: `int`
        `i`th SU
    `joint_angular_velocity`: `float`
        Angular velocity
    'max_angular_velocity': 'float'
        Maximum angular velocity
    `current_time`: `float`
        Current Time
    `method`: `str`
        Determines if we are using `analytical`, `normal_mittendorfer` or `mittendorfer`
        methods (which we modified due to some possible missing terms).
    """
    methods = ['analytical', 'mittendorfer', 'normal_mittendorfer']
    if method not in methods:
        raise ValueError(f'There is no method called {method}\n' +
                         f'Please Choose from {methods}')

    rs_T_su = kinematic_chain.compute_su_TM(
        i_su=i_su, pose_type='current')

    dof_T_su = kinematic_chain.compute_su_TM(
        start_joint=i_rotate_joint,
        i_su=i_su,
        pose_type='current')

    dof_w_su = np.array([0, 0, joint_angular_velocity])
    dof_alpha_su = np.array([0, 0, joint_angular_acceleration])
    su_g, su_Ac, su_At = compute_acceleration_analytically(
        inert_w_body=dof_w_su,
        inert_r_body=dof_T_su.position,
        inert_alpha_body=dof_alpha_su,
        body_R_inert=dof_T_su.R.T,
        body_R_world=rs_T_su.R.T,
        coordinate='body')

    if method == 'analytical':
        return su_g + su_Ac + su_At

    # The following will run if method is mittendorfer's method
    su_At = compute_tangential_acceleration_numerically(
        kinematic_chain=kinematic_chain,
        i_rotate_joint=i_rotate_joint,
        i_su=i_su,
        joint_angular_velocity=joint_angular_velocity,
        max_angular_velocity=max_angular_velocity,
        current_time=current_time)

    if method == 'normal_mittendorfer':
        # return su_At
        return su_g + su_At

    # Every joint rotates along its own z axis, one joint moves at a time
    return su_g + su_Ac + su_At


def centripetal_acceleration(r, w):
    r"""
    .. math:: `a = \omega \times \omega \times r`

    Arguments
    ------------
    `r`: `float`
        Position vector r of the body in the inertial coordinate
    `w`: `float`
        Angular Velocity of an axis in the inertial coordinate.
    """
    # a = w x w x r
    return np.cross(w, np.cross(w, r))


def tangential_acceleration(r, alpha):
    """
    .. math:: `a = \alpha \times r`

    Arguments
    ----------
    `r`: `float`
        Position vector r of the body in the inertial coordinate
    `alpha`: `float`
        Angular Acceleration of an axis in the inertial coordinate.
    """
    # a = al x r
    return np.cross(alpha, r)


def compute_acceleration_analytically(inert_w_body, inert_r_body, inert_alpha_body,
                                      body_R_inert=None, body_R_world=None, inert_R_world=None,
                                      coordinate='body'):
    """
    There are 3 coordinates to remember.
    1. World Frame (Fixed to the world)
    2. Inertial Frame (each Rotating Joint Coordinate)
    3. Body Frame (SU Coordinate)

    We use the following notation `coord1_variable_coord2`.
    This represents the variable of coord2 defined in coord1.

    For example, SU's linear accelerations can be represented as rs_a_su,
    which means acceleration a measured in SU coordinate defined in the world coordinate.

    We use the same notation for the rotation matrix `coord1_R_coord2`,
    but this represents that it rotations some variables from `coord2` to `coord1`.

    For example, if you want the gravity in the su coordinate,
    su_g = su_R_rs * rs_g
    """
    world_g = np.array([0, 0, 9.80])
    inert_Ac_body = centripetal_acceleration(r=inert_r_body, w=inert_w_body)
    inert_At_body = tangential_acceleration(r=inert_r_body, alpha=inert_alpha_body)

    if coordinate == 'body':
        if body_R_inert is None or body_R_world is None:
            raise ValueError('You must provide Rotation matrices body_R_inert and body_R_world')
        # Convert to body coordinate
        body_g = np.dot(body_R_world, world_g)
        body_Ac = np.dot(body_R_inert, inert_Ac_body)
        body_At = np.dot(body_R_inert, inert_At_body)
        return body_g, body_Ac, body_At

    elif coordinate == 'world':
        if inert_R_world is None:
            raise ValueError('You must provide a Rotation Matrix inert_R_world')
        # Convert to world coordinate
        world_Ac = np.dot(inert_R_world.T, inert_Ac_body)
        world_At = np.dot(inert_R_world.T, inert_At_body)
        return world_g, world_Ac, world_At

    else:
        raise KeyError(f'Coordinate name "{coordinate}" is invalid\n' +
                       'Please choose from "body", "inertial", or "world"')


def compute_2nd_order_derivative(x_func, t=0, dt=0.001):
    # dt should be small value, recommended to use 1/(1000 * freq)
    return ((x_func(t+dt) + x_func(t-dt) - 2*x_func(t)) / (dt**2))


def compute_tangential_acceleration_numerically(kinematic_chain, i_rotate_joint, i_su,
                                                joint_angular_velocity, max_angular_velocity,
                                                current_time):
    """
    Returns tangential acceleration in RS coordinate.
    The acceleration is computed by taking 2nd derivative of the position.
    This small change in position in Rotating Coordinate is only in the
    tangential direction. Thus, you can only compute the tangential acceleration,
    from this method.


    Arguments
    ---------
    `kinematic_chain`: `robotic_skin.calibration.kinematic_chain.KinematicChain`
        Robot's Kinematic Chain
    'i_rotate_joint': 'int'
        dof 'd'
    `i`: `int`
        imu `i`
    'max_angular_velocity': 'float'
        Maximum angular velocity
    `current_time`: float`
        Current Time
    """
    def current_su_transformation_matrix(angle):
        """
        Returns position of the current skin unit

        Arguments
        ----------
        `t`: `float`
            Given Time t
        """
        dof_T_dof, rs_T_dof = kinematic_chain.get_current_TMs()
        kinematic_chain.add_a_pose(
            i_joint=i_rotate_joint,
            pose=angle,
            dof_T_dof=dof_T_dof,
            rs_T_dof=rs_T_dof)
        return kinematic_chain._compute_su_TM(
            i_su=i_su,
            dof_T_dof=dof_T_dof,
            rs_T_dof=rs_T_dof,
            start_joint=i_rotate_joint)

    def su_position_in_sinmotion(t):
        angle = (max_angular_velocity / (2*np.pi*C.PATTERN_FREQ)) * (1 - np.cos(2*np.pi*C.PATTERN_FREQ * t))
        T = current_su_transformation_matrix(angle)
        return T.position

    # Compute the Acceleration from 3 close positions
    dof_A = compute_2nd_order_derivative(x_func=su_position_in_sinmotion, t=current_time)
    # Compute current Angle during sinuosoidal motion
    angle = (max_angular_velocity / (2*np.pi*C.PATTERN_FREQ)) * (1 - np.cos(2*np.pi*C.PATTERN_FREQ * current_time))
    # Get current transformation matrix of SU defined in dof coordinate
    dof_T_su = current_su_transformation_matrix(angle=angle)
    # Tangential Vector (to the circular motion)
    e_t = np.cross([0, 0, 1], dof_T_su.position)
    e_t = e_t / np.linalg.norm(e_t)

    # Only retrieve the tangential element of dof_A,
    # because dof_A also includes unnecessary centripetal acceleration
    dof_At = e_t * np.dot(e_t, dof_A)
    su_At = np.dot(dof_T_su.R.T, dof_At)

    # plot_projected_acceleration(dof_A, e_r, e_t, dof_At, dof_Ac)
    return su_At


def plot_projected_acceleration(dof_A, e_r, e_t, dof_At, dof_Ac):
    origin = np.zeros(2)
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    circle = plt.Circle((0, 0), 1.0, color='black', fill=False)

    radius = np.vstack((origin, e_r[:2]))
    tangent = np.vstack((e_r[:2], e_r[:2] + e_t[:2]))
    accelerations = np.vstack((e_r[:2], e_r[:2] + dof_A[:2]))
    At = np.vstack((e_r[:2], e_r[:2] + dof_At[:2]))
    Ac = np.vstack((e_r[:2], e_r[:2] + dof_Ac[:2]))

    ax.plot(At[:, 0], At[:, 1], label='At')
    ax.plot(Ac[:, 0], Ac[:, 1], label='Ac')
    ax.plot(radius[:, 0], radius[:, 1], label='e_r', color='grey', lw=1, alpha=0.7)
    ax.plot(tangent[:, 0], tangent[:, 1], label='e_t', color='grey', lw=1, alpha=0.7)
    ax.plot(accelerations[:, 0], accelerations[:, 1], label='estimated accelerations')
    ax.add_artist(circle)
    ax.axis('equal')
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.legend()

    plt.show()


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
                    angular_velocity = angular_velocities[idx]

                    # TODO: parse start_joint. Currently, there is a bug
                    kinematic_chain.set_poses(poses, end_joint=i_joint)
                    model_accel = estimate_acceleration(kinematic_chain=kinematic_chain,
                                                        i_rotate_joint=d_joint,
                                                        i_su=i_su,
                                                        joint_angular_velocity=angular_velocity)

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
                data = self.data.dynamic[self.pose_names[p]][self.joint_names[d_joint]][self.imu_names[i_su]]

                max_accel_trains = data[:, :3]
                joints = data[:, 3:10]
                times = data[:, 10]
                joint_angular_accelerations = data[:, 11]
                max_angular_velocity = data[0, 12]
                joint_angular_velocities = data[:, 13]

                n_eval = 4
                for i in range(n_eval):
                    n_data = data.shape[0]
                    if n_data <= i:
                        break

                    idx = i*int(n_data/n_eval)
                    max_accel_train = max_accel_trains[idx, :]
                    poses = joints[idx, :]
                    time = times[idx]
                    joint_angular_acceleration = joint_angular_accelerations[idx]
                    joint_angular_velocity = joint_angular_velocities[idx]

                    # kinematic_chain.set_poses(joints)
                    kinematic_chain.set_poses(poses, end_joint=i_joint)
                    # use mittendorfer's original or modified based on condition
                    max_accel_model = estimate_acceleration(kinematic_chain=kinematic_chain,
                                                            i_rotate_joint=d_joint,
                                                            i_su=i_su,
                                                            joint_angular_velocity=joint_angular_velocity,
                                                            joint_angular_acceleration=joint_angular_acceleration,
                                                            max_angular_velocity=max_angular_velocity,
                                                            current_time=time,
                                                            method=self.method)

                    # logging.debug(f'[Pose{p}, Joint{d_joint}, SU{i_su}@Joint{i_joint}]\t' +
                    #             f'Model: {n2s(max_accel_model, 4)} SU: {n2s(max_accel_train, 4)}')
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
