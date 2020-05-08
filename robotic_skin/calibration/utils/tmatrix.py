import numpy as np


class TransformationMatrix():
    """
    Class for Transformation Matrix
    Manages all its parameters and computation
    It also outputs Rotation Matrix and Position of the transformed result
    """
    def __init__(self, theta=None, d=None, a=None, alpha=None):
        """
        Constructor for TransformationMatrix.
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
    def from_numpy(cls, params, keys=['theta', 'd', 'a', 'alpha']):
        n_params = len(keys)
        if not isinstance(params, np.ndarray):
            raise ValueError("'params' should be a np.array")
        if params.size != n_params:
            raise ValueError("Size of 'params' should be %i" % (n_params))

        d = {k: v for k, v in zip(keys, params)}
        return cls.from_dict(d)

    @classmethod
    def from_list(cls, params, keys=['theta', 'd', 'a', 'alpha']):
        return cls.from_numpy(np.array(params), keys)

    @classmethod
    def from_bounds(cls, bounds, keys=['theta', 'd', 'a', 'alpha']):
        n_row = len(keys)
        if not isinstance(bounds, np.ndarray):
            raise ValueError("'bounds' should be a np.ndarray")
        if bounds.shape != (n_row, 2):
            raise ValueError("Shape of 'bounds' should be (%i, 2)" % (n_row))

        params = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1])
        d = {k: v for k, v in zip(keys, params)}
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
        T: TransformationMatrix
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

    def set_params(self, params, keys=['theta', 'd', 'a', 'alpha']):
        """
        Set parameters that have been optimized

        Parameters
        -----------
        params: np.array
            DH parameters
        """
        T = TransformationMatrix.from_numpy(params, keys)
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

