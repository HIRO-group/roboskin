"""
Utilities module for Robotic Skin.
"""
import os
import yaml
import numpy as np
import pyquaternion as pyqt
from geometry_msgs.msg import Quaternion
import math


class TransformationMatrix():
    """
    Class for Transformation Matrix
    Manages all its parameters and computation
    It also outputs Rotation Matrix and Position of the transformed result
    """
    def __init__(self, theta=None, d=None, a=None, alpha=None):
        """
        Constructor for TransMat.
        It creates a transformation matrix from DH Parameters

        Parameters
        ------------
        params: np.array
            DH parameters
            It includes theta, d, a, alpha
            For DH Parameters, please refer to this video
            https://robotacademy.net.au/lesson/denavit-hartenberg-notation/
        """
        params = np.array([theta, d, a, alpha], dtype=float)
        # Only select provided keys (which are not None)
        self.key_index = np.argwhere(~np.isnan(params)).flatten()
        self.params = np.nan_to_num(params)
        self.matrix = self.transformation_matrix(*self.params)

    def transformation_matrix(self, th, d, a, al):
        """

        Create a transformation matrix
        DH Parameters are defined with only 4 parameters.
        2 Translational parameters and 2 Rotations parameters.
        Here, we follow the "Modified DH Parameter" notation and
        not the original classic DH Parameter invnted by Denavit and Hartenberg.

        From (n-1)th coordinate frame,
        1. Rotate for al [rad] around x axis (Rx).
        2. Displace for a [m] along x axis (Tx).
        3. Rotate for th [rad] around z axis (Rz)
        4. Displace for d [m] along z axis (Tz)
        to get to the nth coordinate frame in this order.

        ..math:: {}^{n-1}_{n}T = Rx * Tx * Rz * Tz
        ..math::
            \left[
            \begin{array}{c|c}
                {}^{n-1}_{n} R & {}^{n-1}_{n}P \\
                \hline
                0 & 1
            \end{array}
            \right]

            The superscript represents which frame the variable is in,
            and the subscript represents from which frame the variable is stated.
            For example, :math:`{}^{0}_{1}P` represents the position of the 1st link (frame)
            in the the world frame 0. So if you want to compute the end-effector's position
            in the world frame, you write as :math:`{}^{0}_{6}P`.

            If you want to rotate the gravity vector from world (frame 0) to SU 6,
            you write as :math:`{}^{SU_6}g = {}^{SU_6}_{0}R * {}^{0}g`.
            You can compute :math:`{}^{SU_6}_{0}R`  by,

            ..math::
                {}^{0}_{SU_6}T = {}^{0}_{1}T * {}^{1}_{2}T * ... * {}^{6}_{SU_6}T
                {}^{SU_6}_{0}R = ({}^{0}_{SU_6}T).R.T

        Note that
            Rz * Tz = Tz * Rz
            Rx * Tx = Tx * Rx

        Source:
        - http://www4.cs.umanitoba.ca/~jacky/Robotics/Papers/spong_kinematics.pdf
        - https://en.wikipedia.org/wiki/Denavit%E2%80%93Hartenberg_parameters#Modified_DH_parameters

        Parameters
        ------------
        th:
            Rotation theta around z axis (rad)
        d:
            Displacement relative to z axis (m)
        a:
            Displacement relative to x axis (m)
        al:
            Rotation alpha around x axis (rad)

        Returns
        ---------
        np.ndarray
            transformation matrix
            returns 4x4 matrix of the form
        """  # noqa: W605
        """
        Classic DH Parameter Transformation
        ..math:: {}^{n-1}_{n}T = Tz * Rz * Tx * Rz

        return np.array([
            [np.cos(th), -np.sin(th)*np.cos(al), np.sin(th)*np.sin(al), a*np.cos(th)],
            [np.sin(th), np.cos(th)*np.cos(al), -np.cos(th)*np.sin(al), a*np.sin(th)],
            [0, np.sin(al), np.cos(al), d],
            [0, 0, 0, 1]
        ])
        """
        return np.array([
            [np.cos(th), -np.sin(th), 0, a],
            [np.sin(th)*np.cos(al), np.cos(th)*np.cos(al), -np.sin(al), -d*np.sin(al)],
            [np.sin(th)*np.sin(al), np.cos(th)*np.sin(al), np.cos(al), d*np.cos(al)],
            [0, 0, 0, 1],
        ])

    @classmethod
    def from_dict(cls, params_dict):
        return cls(**params_dict)

    @classmethod
    def from_numpy(cls, params, key_order=['theta', 'd', 'a', 'alpha']):
        n_params = len(key_order)
        if not isinstance(params, np.ndarray):
            raise ValueError("'params' should be a np.array")
        if params.size != n_params:
            raise ValueError("Size of 'params' should be %i" % (n_params))

        d = {k: v for k, v in zip(key_order, params)}
        return cls.from_dict(d)

    @classmethod
    def from_list(cls, params, key_order=['theta', 'd', 'a', 'alpha']):
        return cls.from_numpy(np.array(params), key_order)

    @classmethod
    def from_bounds(cls, bounds, key_order=['theta', 'd', 'a', 'alpha']):
        n_row = len(key_order)
        if not isinstance(bounds, np.ndarray):
            raise ValueError("'bounds' should be a np.ndarray")
        if bounds.shape != (n_row, 2):
            raise ValueError("Shape of 'bounds' should be (%i, 2)" % (n_row))

        params = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1])
        d = {k: v for k, v in zip(key_order, params)}
        return cls.from_dict(d)

    @classmethod
    def from_matrix(cls, matrix):
        T = cls()
        T.matrix = matrix
        return T

    def __mul__(self, T):
        """
        In our implementation, we use * for dot product

        Parameters
        ------------
        mat: np.ndarray
            4 by 4 ndarray transformation matrix

        Returns
        ----------
        T: TransMat
            Resulting transformation matrix from dot products
            of two transformation matrices
        """
        new_matrix = np.dot(self.matrix, T.matrix)
        return TransformationMatrix.from_matrix(new_matrix)

    @property
    def R(self):
        """
        Rotation Matrix

        Returns
        ----------
        np.ndarray
            Rotation Matrix
        """
        return self.matrix[:3, :3]

    @property
    def q(self):
        """
        Quaternion as a result of the transformation
        """
        q = pyqt.Quaternion(matrix=self.R)
        return pyquat_to_numpy(q)

    @property
    def position(self):
        """
        Position as a result of the transformation

        Returns
        -----------
        np.ndarray
            Position of the resulting transformation
        """
        return self.matrix[:3, 3]

    def set_params(self, params, **kwargs):
        """
        Set parameters that have been optimized

        Parameters
        -----------
        params: np.array
            DH parameters
        """
        T = TransformationMatrix.from_numpy(params, kwargs)
        self.matrix = np.copy(T.matrix)

    @property
    def parameters(self):
        """
        Returns
        ---------
        np.array
            DH parameters of this transformation matrix
        """
        return self.params[self.key_index]

    def __str__(self):
        return np.array2string(self.matrix)


class TransMat():
    """
    Class for Transformation Matrix
    Manages all its parameters and computation
    It also outputs Rotation Matrix and Position of the transformed result
    """
    def __init__(self, params=None, bounds=None, mat=None):
        """
        Constructor for TransMat.
        It creates a transformation matrix from DH Parameters

        Parameters
        ------------
        params: np.array
            DH parameters
            It includes theta, d, a, alpha
            For DH Parameters, please refer to this video
            https://robotacademy.net.au/lesson/denavit-hartenberg-notation/
        """
        # if nothing is provided, set to zeros
        if params is None:
            params = np.zeros(4)
        # if bounds are provided, apply to params
        if bounds is not None:
            params = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1])

        params = np.array(params)
        self.n_params = params.size

        # only 1,2 and 4 parameters are allowed
        if self.n_params not in [1, 2, 4]:
            raise ValueError('Please provide a valide number of parameters. None, 1, 2 or 4')

        th, d, a, al = self.check_params(params)
        self.params = np.array([th, d, a, al])
        self.mat = self.transformation_matrix(th, d, a, al)

        # If Matrix is always computed and given
        if mat is not None:
            self.mat = mat
            self.params = None
            self.n_params = None

    def check_params(self, params):
        """
        Check the size of the given dh parameters
        Complement other parameters according to the size

        Parameters
        -----------
        params: np.array
            DH Parameters
        """
        if params.size == 1:
            th = params
            d, a, al = 0.0, 0.0, 0.0
        elif params.size == 2:
            th, d = params
            a, al = 0.0, 0.0
        elif params.size == 4:
            th, d, a, al = params
        else:
            raise ValueError('Wrong number of parameters passed. It should be 1, 2 or 4')

        return th, d, a, al

    def transformation_matrix(self, th, d, a, al):
        """

        Create a transformation matrix
        DH Parameters are defined with only 4 parameters.
        2 Translational parameters and 2 Rotations parameters.
        Here, we follow the "Modified DH Parameter" notation and
        not the original classic DH Parameter invnted by Denavit and Hartenberg.

        From (n-1)th coordinate frame,
        1. Rotate for al [rad] around x axis (Rx).
        2. Displace for a [m] along x axis (Tx).
        3. Rotate for th [rad] around z axis (Rz)
        4. Displace for d [m] along z axis (Tz)
        to get to the nth coordinate frame in this order.

        ..math:: {}^{n-1}_{n}T = Rx * Tx * Rz * Tz
        ..math::
            \left[
            \begin{array}{c|c}
                {}^{n-1}_{n} R & {}^{n-1}_{n}P \\
                \hline
                0 & 1
            \end{array}
            \right]

            The superscript represents which frame the variable is in,
            and the subscript represents from which frame the variable is stated.
            For example, :math:`{}^{0}_{1}P` represents the position of the 1st link (frame)
            in the the world frame 0. So if you want to compute the end-effector's position
            in the world frame, you write as :math:`{}^{0}_{6}P`.

            If you want to rotate the gravity vector from world (frame 0) to SU 6,
            you write as :math:`{}^{SU_6}g = {}^{SU_6}_{0}R * {}^{0}g`.
            You can compute :math:`{}^{SU_6}_{0}R`  by,

            ..math::
                {}^{0}_{SU_6}T = {}^{0}_{1}T * {}^{1}_{2}T * ... * {}^{6}_{SU_6}T
                {}^{SU_6}_{0}R = ({}^{0}_{SU_6}T).R.T

        Note that
            Rz * Tz = Tz * Rz
            Rx * Tx = Tx * Rx

        Source:
        - http://www4.cs.umanitoba.ca/~jacky/Robotics/Papers/spong_kinematics.pdf
        - https://en.wikipedia.org/wiki/Denavit%E2%80%93Hartenberg_parameters#Modified_DH_parameters

        Parameters
        ------------
        th:
            Rotation theta around z axis (rad)
        d:
            Displacement relative to z axis (m)
        a:
            Displacement relative to x axis (m)
        al:
            Rotation alpha around x axis (rad)

        Returns
        ---------
        np.ndarray
            transformation matrix
            returns 4x4 matrix of the form
        """  # noqa: W605
        """
        Classic DH Parameter Transformation
        ..math:: {}^{n-1}_{n}T = Tz * Rz * Tx * Rz

        return np.array([
            [np.cos(th), -np.sin(th)*np.cos(al), np.sin(th)*np.sin(al), a*np.cos(th)],
            [np.sin(th), np.cos(th)*np.cos(al), -np.cos(th)*np.sin(al), a*np.sin(th)],
            [0, np.sin(al), np.cos(al), d],
            [0, 0, 0, 1]
        ])
        """
        return np.array([
            [np.cos(th), -np.sin(th), 0, a],
            [np.sin(th)*np.cos(al), np.cos(th)*np.cos(al), -np.sin(al), -d*np.sin(al)],
            [np.sin(th)*np.sin(al), np.cos(th)*np.sin(al), np.cos(al), d*np.cos(al)],
            [0, 0, 0, 1],
        ])

    def dot(self, T):
        """
        In our implementation, we use * for dot product

        Parameters
        ------------
        mat: np.ndarray
            4 by 4 ndarray transformation matrix

        Returns
        ----------
        T: TransMat
            Resulting transformation matrix from dot products
            of two transformation matrices
        """
        new_mat = np.dot(self.mat, T.mat)
        return TransMat(mat=new_mat)

    @property
    def R(self):
        """
        Rotation Matrix

        Returns
        ----------
        np.ndarray
            Rotation Matrix
        """
        return self.mat[:3, :3]

    @property
    def q(self):
        """
        Quaternion as a result of the transformation
        """
        q = pyqt.Quaternion(matrix=self.R)
        return pyquat_to_numpy(q)

    @property
    def position(self):
        """
        Position as a result of the transformation

        Returns
        -----------
        np.ndarray
            Position of the resulting transformation
        """
        return self.mat[:3, 3]

    def set_params(self, params):
        """
        Set parameters that have been optimized

        Parameters
        -----------
        params: np.array
            DH parameters
        """
        th, d, a, al = self.check_params(params)
        self.params = np.array([th, d, a, al])
        self.mat = self.transformation_matrix(th, d, a, al)

    @property
    def parameters(self):
        """
        Returns
        ---------
        np.array
            DH parameters of this transformation matrix
        """
        if self.n_params is None:
            return None
        elif self.n_params == 1:
            return self.params[0]
        elif self.n_params == 2:
            return self.params[[0, 1]]
        else:
            return self.params

    def __str__(self):
        return np.array2string(self.mat)


class ParameterManager():
    """
    Class for managing DH parameters
    """
    def __init__(self, n_joint, bounds, bounds_su, dhparams=None):
        """
        TODO For now, we assume n_sensor is equal to n_joint

        Arguments
        -----------
        n_joints: int
            Number of joints
        bounds: np.ndarray
            Bounds for DH parameters
        """
        self.n_joint = n_joint
        self.bounds = bounds
        self.bounds_su = bounds_su
        self.dhparams = dhparams

        if self.dhparams is None:
            # 10 parameters to optimize.
            # uninitialized dh params
            self.Tdof2dof = [TransMat(bounds=bounds) for i in range(n_joint)]
        else:
            # 6 parameters to optimize.
            self.Tdof2dof = [TransMat(dhparams['joint' + str(i+1)]) for i in range(n_joint)]

        self.Tdof2vdof = [TransMat(bounds=bounds_su[:2, :]) for i in range(n_joint)]
        self.Tvdof2su = [TransMat(bounds=bounds_su[2:, :]) for i in range(n_joint)]

    def get_params_at(self, i):
        """
        if n_joint is 7 DoF i = 0, 1, ..., 6

        Arguments
        ---------------
        i: int
            ith joint (ith sensor)

        Returns
        --------
        params: np.array
            Next DH parameters to be optimized
        """
        if self.dhparams is not None:
            # optimizing just su dh params.
            params = np.r_[self.Tdof2vdof[i].parameters, self.Tvdof2su[i].parameters]
            bounds = self.bounds_su[:, :]

            assert params.size == 6
            assert bounds.shape == (6, 2)
        else:
            # optimizing all dh parameters
            params = np.r_[self.Tdof2dof[i].parameters,
                           self.Tdof2vdof[i].parameters,
                           self.Tvdof2su[i].parameters]
            bounds = np.vstack((self.bounds[:, :], self.bounds_su[:, :]))

            assert params.size == 10
            assert bounds.shape == (10, 2)

        return params, bounds

    def get_tmat_until(self, i):
        """
        get transformation matrices when optimizing ith joint (sensor)

        Arguments
        ----------
        i: int
            ith joint (sensor)

        Returns
        --------
        list of TransMat
            Transformation Matrices between DoFs
        list of TransMat
            Transformation Rotation Matrices for all joints
        """
        if self.dhparams is not None:
            return self.Tdof2dof[:i+1]
        else:
            return self.Tdof2dof[:max(0, i+1)]

    def set_params_at(self, i, params):
        """
        Set DH parameters
        Depending of if we
        are optimizing 6 (just su params)
        or 10 (all dh params)

        Arguments
        ------------
        int: i
            ith joint (sensor)
        parmas: np.array
            DH Parameters
        """
        if self.dhparams is not None:
            self.Tdof2vdof[i].set_params(params[:2])
            self.Tvdof2su[i].set_params(params[2:])
        else:
            self.Tdof2dof[i].set_params(params[:4])
            self.Tdof2vdof[i].set_params(params[4:6])
            self.Tvdof2su[i].set_params(params[6:])


def tfquat_to_pyquat(q):
    """
    Converts a tf quaternion to a pyqt quaternion
    """
    return pyqt.Quaternion(w=q.w, x=q.x, y=q.y, z=q.z)


def pyquat_to_tfquat(q):
    """
    Converts a pyqt quaternion to a tf quaternion
    """
    q = q.elements
    return Quaternion(q[1], q[2], q[3], q[0])


def pyquat_to_numpy(q):
    """
    Convert a pyqt quaternion to a numpy array.
    """
    q = q.elements
    return np.array([q[1], q[2], q[3], q[0]])


def quaternion_l2_distance(q1, q2):
    """
    A metric for computing the distance
    between 2 quaternions.
    sources:
    - https://fgiesen.wordpress.com/2013/01/07/small-note-on-quaternion-distance-metrics/
    - http://kieranwynn.github.io/pyquaternion/#accessing-individual-elements
    """
    return 2*(1 - np.dot(q1.elements, q2.elements))


def quaternion_from_two_vectors(source, target):
    """
    Computes the quaternion from vector `source`
    to vector `target`.

    """
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)

    axis = np.cross(source, target)
    costh = np.dot(source, target)

    angle = np.arccos(costh)

    if angle == 0.0:
        return pyqt.Quaternion()

    return pyqt.Quaternion(axis=axis, angle=angle)


def angle_between_quaternions(q_1: np.ndarray, q_2: np.ndarray, output_in_degrees: bool = False) -> float:  # noqa: E999
    r"""
    Angle between quaternions a and b in degrees. Please note the input quaternions should be of
    form np.ndarray([x, y, z, w]).
    The formula for angle between quaternions is:

    .. math::
        \theta = \cos^{-1}\bigl(2\langle q_1,q_2\rangle^2 -1\bigr)

    where ⟨q1,q2⟩ denotes the inner product of the corresponding quaternions:

    .. math::
        \langle a_1 +b_1 \textbf{i} + c_1 \textbf{j} + d_1 \textbf{k}, \\
        a_2 + b_2 \textbf{i} + c_2 \textbf{j} + d_2 \textbf{k}\rangle \\
        = a_1a_2 + b_1b_2 + c_1 c_2 + d_1d_2.

    Reference: https://math.stackexchange.com/questions/90081/quaternion-distance
    :param q_1: np.ndarray
        Quaternion a
    :param q_2: np.ndarray
        Quaternion b
    :return: float
        Angle between quaternions in degrees
    """
    if not (math.isclose(np.linalg.norm(q_1), 1.0, abs_tol=0.1) and math.isclose(np.linalg.norm(q_2), 1.0, abs_tol=0.1)):
        raise Exception("Please only pass unit quaternions")
    angle = np.arccos(2 * ((q_1 @ q_2) ** 2) - 1)  # noqa: E999
    if np.isnan(angle):
        # usually some values overflow 1. arc-cos isn't defined in range > 1
        # So it's just an overflow error and the angle can be safely be assumed zero
        return 0.0
    if output_in_degrees:
        angle_in_degrees = (angle / np.pi) * 180
        return angle_in_degrees
    return angle


def n2s(x, precision=2):
    """
    converts numpy array to string.

    Arguments
    ---------
    `x`: `np.array`
        The numpy array to convert to a string.

    `precision`: `int`
        The precision desired on each entry in the array.

    """
    return np.array2string(x, precision=precision, separator=',', suppress_small=True)


def get_IMU_pose(Tdofs, Tdof2su, joints=None):
    """
    gets the imu pose.

    """
    T = TransMat(np.zeros(4))
    # Transformation Matrix until the joint
    # where SU is attached
    if joints is not None:
        for Tdof, j in zip(Tdofs, joints):
            T = T.dot(Tdof).dot(TransMat(j))
    else:
        for Tdof in Tdofs:
            T = T.dot(Tdof)
    # Transformation Matrix until SU
    T = T.dot(Tdof2su)

    return T.position, T.q


def load_robot_configs(configdir, robot):
    """
    Loads robot's DH parameters, SUs' DH parameters and their poses

    configdir: str
        Path to the config directory where robot yaml files exist
    robot: str
        Name of the robot
    """
    filepath = os.path.join(configdir, robot + '.yaml')
    try:
        with open(filepath) as file:
            return yaml.load(file, Loader=yaml.FullLoader)
    except Exception:
        raise ValueError('Please provide a valid config directory with robot yaml files')
