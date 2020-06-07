import numpy as np
import torch
import pyquaternion as pyqt
from robotic_skin.calibration.utils.quaternion import pyqt_to_np


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
        # if parameters are tensors, don't convert to np array
        params = np.array([theta, d, a, alpha], dtype=float)
        self.key_index = np.argwhere(~np.isnan(params)).flatten()
        self.params = np.nan_to_num(params)

        qx = pyqt.Quaternion(axis=[1, 0, 0], angle=self.params[3])
        qz = pyqt.Quaternion(axis=[0, 0, 1], angle=self.params[0])

        if type(theta) == torch.Tensor:
            if theta is None:
                theta = torch.tensor(0.).double().cuda()
            if d is None:
                d = torch.tensor(0.).double().cuda()
            if a is None:
                a = torch.tensor(0.).double().cuda()
            if alpha is None:
                alpha = torch.tensor(0.).double().cuda()

            self.params = torch.cat((theta.view(-1), d.view(-1), a.view(-1), alpha.view(-1)))
            self.is_tensor = True
        else:
            self.is_tensor = False
            # Only select provided keys (which are not None)

        self.matrix = self.transformation_matrix(*self.params)
        self.q = qx * qz

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
        if self.is_tensor:
            # first concatenate to each row
            # then stack for the whole transformation matrix.
            x1 = torch.cat((torch.cos(th).view(-1), -torch.sin(th).view(-1),
                            torch.tensor(0).double().cuda().view(-1), a.view(-1)))
            x2 = torch.cat(((torch.sin(th)*torch.cos(al)).view(-1),
                            (torch.cos(th)*torch.cos(al)).view(-1),
                            -torch.sin(al).view(-1),
                            ((-d*torch.sin(al)).view(-1))))
            x3 = torch.cat(((torch.sin(th)*torch.sin(al)).view(-1),
                            (torch.cos(th)*torch.sin(al)).view(-1),
                            (torch.cos(al)).view(-1),
                            (d*torch.cos(al)).view(-1)))
            x4 = torch.tensor([0, 0, 0, 1]).double().cuda()
            mat = torch.stack((x1, x2, x3, x4))
        else:
            mat = np.array([
                [np.cos(th), -np.sin(th), 0, a],
                [np.sin(th)*np.cos(al), np.cos(th)*np.cos(al), -np.sin(al), -d*np.sin(al)],
                [np.sin(th)*np.sin(al), np.cos(th)*np.sin(al), np.cos(al), d*np.cos(al)],
                [0, 0, 0, 1],
            ])
            if self.is_tensor:
                return torch.from_numpy(mat).double().cuda().requires_grad_(True)
                # standard np array.
        return mat

    @classmethod
    def from_dict(cls, params_dict):
        return cls(**params_dict)

    @classmethod
    def from_numpy(cls, params, keys=['theta', 'd', 'a', 'alpha']):
        n_params = len(keys)
        if not isinstance(params, np.ndarray) and not isinstance(params, torch.Tensor):
            raise ValueError("'params' should be a np.array or torch.Tensor.")
        if params.shape[0] != n_params:
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
        new_matrix = self.mm_fnc(self.matrix, T.matrix)
        new_q = self.q * T.q
        T = TransformationMatrix()
        if self.is_tensor:
            T.tensor_()
        T.matrix = new_matrix
        T.q = new_q
        return T

    def __call__(self, theta):
        params = self.copy_fnc(self.params)
        params[0] += theta
        T = TransformationMatrix(*params)
        if self.is_tensor:
            T.tensor_()
        q = pyqt.Quaternion(axis=[0, 0, 1], angle=theta)
        T.q = self.q * q
        return T

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
    def quaternion(self):
        """
        Quaternion as a result of the transformation
        """
        return pyqt_to_np(self.q)

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
        self.params = params
        if self.is_tensor:
            T.tensor_()
        self.matrix = self.copy_fnc(T.matrix)

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
        """
        result of print(tmat)
        or
        str(tmat)
        """
        if self.is_tensor:
            return str(self.matrix)
        else:
            return np.array2string(self.matrix)

    def copy_fnc(self, input):
        """
        generic copy function.
        """
        if self.is_tensor:
            return input.clone()
        else:
            return np.copy(input)

    def mm_fnc(self, x1, x2):
        """
        depending on if the matrix was converted to a tensor,
        determine the correct matrix multiplication function.
        """
        if type(x1) != type(x2):
            raise TypeError("Matrices are of different types.")
        if self.is_tensor:
            return torch.mm(x1, x2)
        else:
            return np.dot(x1, x2)

    def tensor_(self):
        """
        converts transformation matrix to a tensor in place.
        """
        self.is_tensor = True
        if type(self.matrix) == torch.Tensor:
            pass
        elif type(self.matrix) == np.ndarray:
            self.params = torch.from_numpy(self.params).double().cuda().requires_grad_(True)
            self.matrix = torch.from_numpy(self.matrix).double().cuda().requires_grad_(True)
        else:
            raise TypeError("self.matrix is of wrong type!")

    def numpy_(self):
        """
        converts transformation matrix to a numpy array in place.
        """
        self.is_tensor = False

        if type(self.matrix) == torch.Tensor:
            self.params = self.params.detach().numpy()
            self.matrix = self.matrix.detach().numpy()
        elif type(self.matrix) == np.ndarray:
            pass
        else:
            raise TypeError("self.matrix is of wrong type!")

    def tensor(self):
        """
        converts transformation matrix to tensor,
        and returns the class.
        """
        self.tensor_()
        return self

    def numpy(self):
        """
        converts transformation matrix to np array,
        and returns the class.
        """
        self.numpy_()
        return self
