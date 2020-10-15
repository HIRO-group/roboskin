import numpy as np
import roboskin.const as C


def estimate_acceleration(kinematic_chain, i_rotate_joint, i_su, method,
                          joint_angular_velocity, joint_angular_acceleration=0,
                          current_time=0, angle_func=None):
    r"""
    Compute an acceleration value from positions.
    .. math:: `a = \frac{f({\Delta t}) + f({\Delta t) - 2 f(0)}{h^2}`

    This equation came from Taylor Expansion to get the second derivative from f(t).
    .. math:: f(t+{\Delta t}) = f(t) + hf^{\prime}(t) + \frac{h^2}{2}f^{\prime\prime}(t)
    .. math:: f(t-{\Delta t}) = f(t) - hf^{\prime}(t) + \frac{h^2}{2}f^{\prime\prime}(t)

    Add both equations and plug t=0 to get the above equation

    Arguments
    ------------
    `kinematic_chain`: `roboskin.calibration.kinematic_chain.KinematicChain`
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
        Determines if we are using `analytical`, `our`, `mittendorfer` or `modified_mittendorfer`
        methods (which we modified due to some possible missing terms).
    """
    methods = ['analytical', 'our', 'mittendorfer', 'modified_mittendorfer']
    if method not in methods:
        raise ValueError(f'There is no method called {method}\n' +
                         f'Please Choose from {methods}')

    if angle_func is None:
        def _angle_func(t, **kwargs):
            return joint_angular_velocity*t
        angle_func = _angle_func

    rs_T_su = kinematic_chain.compute_su_TM(
        i_su=i_su, pose_type='current')

    dof_T_su = kinematic_chain.compute_su_TM(
        start_joint=i_rotate_joint,
        i_su=i_su,
        pose_type='current')

    rs_T_dof = kinematic_chain.compute_joint_TM(
        i_joint=i_rotate_joint,
        pose_type='current')

    # In any joint (DoF) coordinate,
    # the the rotational axis is always pointing its z direction.
    # This is how DH parameters are defined.
    dof_w_su = np.array([0, 0, joint_angular_velocity])
    dof_alpha_su = np.array([0, 0, joint_angular_acceleration])
    # Compute acceleration based on the simple physics
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
    rs_A = compute_acceleration_numerically(
        kinematic_chain=kinematic_chain,
        i_rotate_joint=i_rotate_joint,
        i_su=i_su,
        current_time=current_time,
        angle_func=angle_func,
        method=method)

    if 'mittendorfer' in method:
        su_At = np.dot(rs_T_su.R.T, rs_A)

        if method == 'mittendorfer':
            return su_At

        if method == 'modified_mittendorfer':
            return su_g + su_Ac + su_At

    su_At = remove_centripetal_component(rs_A, rs_T_dof, dof_T_su)

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
    This represents a variable of coord2 defined in coord1.

    For example, SU's linear accelerations can be represented as rs_A_su.
    It represents the acceleration A measured in SU coordinate defined in the world coordinate.
    This is what we can measure from the real IMU as well.

    We use the same notation for the rotation matrix `coord1_R_coord2`,
    but this represents how you can rotate soem vector from `coord2 to `coord1`.

    For example, if you want the gravity in the su coordinate,
    su_g = su_R_rs * rs_g
    As a result, the gravity defined in the RS frame is converted to the SU coordinate.

    The inverse of the Rotation Matrix is its transpose.
    Therefore, one can rotate it back to its original frame by
    rs_g = rs_R_su * su_g = su_R_rs.T * su_g
    """
    world_g = np.array([0, 0, C.GRAVITATIONAL_CONSTANT])
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
    r"""
    Take 2nd order derivative of x_func
    This computation comes from the Taylor Expansion

    .. math::
        x(t+dt) = x(t) + x'(t)*dt + 1/2 * x''(t)*dt^2

        x(t-dt) = x(t) - x'(t)*dt + 1/2 * x''(t)*dt^2

        x(t+dt) + x(t-dt) = 2*x(t) + x''(t)*dt^2

        x''(t) = \frac{x(t+dt) + x(t-dt) - 2*x(t)}{dt^2}
    """
    # dt should be small value, recommended to use 1/(1000 * freq)
    return ((x_func(t+dt) + x_func(t-dt) - 2*x_func(t)) / (dt**2))


def compute_acceleration_numerically(kinematic_chain, i_rotate_joint, i_su,
                                     current_time, angle_func, method):
    """
    Returns tangential acceleration in RS coordinate.
    The acceleration is computed by taking 2nd derivative of the position.
    This small change in position in Rotating Coordinate is only in the
    tangential direction. Thus, you can only compute the tangential acceleration,
    from this method.


    Arguments
    ---------
    `kinematic_chain`: `roboskin.calibration.kinematic_chain.KinematicChain`
        Robot's Kinematic Chain
    'i_rotate_joint': 'int'
        dof 'd'
    `i`: `int`
        imu `i`
    `current_time`: float`
        Current Time
    `angle_func`: function
        A function to compute the current angle at time t
    """
    def current_su_position(t):
        angle = angle_func(t=t, i_joint=i_rotate_joint)
        kinematic_chain.init_temp_TM(i_joint=i_rotate_joint, additional_pose=angle)
        T = kinematic_chain.compute_su_TM(i_su=i_su, pose_type='temp')
        return T.position

    # Compute the Acceleration from 3 close positions
    rs_A = compute_2nd_order_derivative(x_func=current_su_position, t=current_time)

    return rs_A


def remove_centripetal_component(rs_A, rs_T_dof, dof_T_su):
    # Convert rs_A to dof_A
    dof_A = np.dot(rs_T_dof.R.T, rs_A)

    # Compute a tangential vector
    # (Tangential to a circle at time t)
    e_t = np.cross([0, 0, 1], dof_T_su.position)
    e_t = e_t / np.linalg.norm(e_t)

    # Only retrieve the tangential element of dof_A,
    # because dof_A also includes unnecessary centripetal acceleration
    dof_At = e_t * np.dot(e_t, dof_A)
    su_At = np.dot(dof_T_su.R.T, dof_At)

    return su_At
