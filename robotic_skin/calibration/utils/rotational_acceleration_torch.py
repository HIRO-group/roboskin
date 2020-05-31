
import torch

def estimate_acceleration_torch(kinematic_chain, i_rotate_joint, i_su,
                          joint_angular_velocity, joint_angular_acceleration=0,
                          current_time=0, angle_func=None, method='analytical'):
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

    if angle_func is None:
        def _angle_func_torch(t):
            return joint_angular_velocity*t
        angle_func = _angle_func

    rs_T_su = kinematic_chain.compute_su_TM(
        i_su=i_su, pose_type='current')

    dof_T_su = kinematic_chain.compute_su_TM(
        start_joint=i_rotate_joint,
        i_su=i_su,
        pose_type='current')

    # In any joint (DoF) coordinate,
    # the the rotational axis is always pointing its z direction.
    # This is how DH parameters are defined.
    dof_w_su = torch.tensor([0, 0, joint_angular_velocity]).double().cuda()
    dof_alpha_su = torch.tensor([0, 0, joint_angular_acceleration]).double().cuda()
    # Compute acceleration based on the simple physics
    su_g, su_Ac, su_At = compute_acceleration_analytically_torch(
        inert_w_body=dof_w_su,
        inert_r_body=dof_T_su.position,
        inert_alpha_body=dof_alpha_su,
        body_R_inert=dof_T_su.R.T,
        body_R_world=rs_T_su.R.T,
        coordinate='body')

    if method == 'analytical':
        return su_g + su_Ac + su_At

    # The following will run if method is mittendorfer's method
    su_At = compute_tangential_acceleration_numerically_torch(
        kinematic_chain=kinematic_chain,
        i_rotate_joint=i_rotate_joint,
        i_su=i_su,
        current_time=current_time,
        angle_func=angle_func)

    if method == 'normal_mittendorfer':
        # return su_At
        return su_g + su_At

    # Every joint rotates along its own z axis, one joint moves at a time
    return su_g + su_Ac + su_At


def centripetal_acceleration_torch(r, w):
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
    return torch.cross(w, torch.cross(w, r))


def tangential_acceleration_torch(r, alpha):
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
    return torch.cross(alpha, r)


def compute_acceleration_analytically_torch(inert_w_body, inert_r_body, inert_alpha_body,
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
    world_g = torch.tensor([0, 0, 9.80]).double().cuda()
    inert_Ac_body = centripetal_acceleration_torch(r=inert_r_body, w=inert_w_body)
    inert_At_body = tangential_acceleration_torch(r=inert_r_body, alpha=inert_alpha_body)

    if coordinate == 'body':
        if body_R_inert is None or body_R_world is None:
            raise ValueError('You must provide Rotation matrices body_R_inert and body_R_world')
        # Convert to body coordinate
        body_g = torch.mm(body_R_world, world_g.view(3, 1)).view(-1)
        body_Ac = torch.mm(body_R_inert, inert_Ac_body.view(3, 1)).view(-1)
        body_At = torch.mm(body_R_inert, inert_At_body.view(3, 1)).view(-1)
        return body_g, body_Ac, body_At

    elif coordinate == 'world':
        if inert_R_world is None:
            raise ValueError('You must provide a Rotation Matrix inert_R_world')
        # Convert to world coordinate
        world_Ac = torch.mm(inert_R_world.T, inert_Ac_body.view(3, 1)).view(-1)
        world_At = torch.mm(inert_R_world.T, inert_At_body.view(3, 1)).view(-1)
        return world_g, world_Ac, world_At

    else:
        raise ValueError(f'Coordinate name "{coordinate}" is invalid\n' +
                       'Please choose from "body", "inertial", or "world"')


def compute_2nd_order_derivative_torch(x_func, t=0, dt=0.001):
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


def compute_tangential_acceleration_numerically_torch(kinematic_chain, i_rotate_joint, i_su,
                                                current_time, angle_func):
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
    `current_time`: float`
        Current Time
    `angle_func`: function
        A function to compute the current angle at time t
    """
    def current_su_position_torch(t):
        angle = angle_func(t=t)
        kinematic_chain.init_temp_TM(i_joint=i_rotate_joint, additional_pose=angle)
        T = kinematic_chain.compute_su_TM(i_su=i_su, pose_type='temp')
        return T.position

    # Compute the Acceleration from 3 close positions
    rs_A = compute_2nd_order_derivative_torch(x_func=current_su_position_torch, t=current_time)
    # Compute current Angle during sinuosoidal motion
    angle = angle_func(t=current_time)
    # Get current transformation matrix of SU defined in dof coordinate
    kinematic_chain.init_temp_TM(i_joint=i_rotate_joint, additional_pose=angle)
    dof_T_su = kinematic_chain.compute_su_TM(i_su=i_su, pose_type='temp', start_joint=i_rotate_joint)
    # dof defined in rs coordinate
    rs_T_dof = kinematic_chain.compute_joint_TM(i_joint=i_rotate_joint, pose_type='temp')
    # Convert rs_A to dof_A
    dof_A = torch.mm(rs_T_dof.R.T, rs_A.view(3, 1)).view(-1)

    # Compute a tangential vector
    # (Tangential to a circle at time t)
    e_t = torch.cross([0, 0, 1], dof_T_su.position)
    e_t = e_t / torch.norm(e_t)

    # Only retrieve the tangential element of dof_A,
    # because dof_A also includes unnecessary centripetal acceleration
    dof_At = e_t * torch.mm(e_t, dof_A.view(3, 1)).view(-1)
    su_At = torch.mm(dof_T_su.R.T, dof_At.view(3, 1)).view(-1)

    return su_At
