"""
Utilities module for Robotic Skin.
"""
import numpy as np 
from pyquaternion import Quaternion

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
        if params is None:
            params = np.zeros(4)

        if bounds is not None: 
            params = np.array([np.random.rand()*(high-low) + low for low, high in bounds])

        params = np.array(params)
        self.n_params = params.size 
        
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
            print(params)
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
        """
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
        if self.n_params == None:
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

        if self.dhparams is not None:
            self.Tdof2dof = [TransMat(dhparams[i, :]) for i in range(n_joint)]
        else:
            self.Tdof2dof = [TransMat(bounds=bounds) for i in range(n_joint)]

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
            params = np.r_[self.Tdof2vdof[i].parameters, self.Tvdof2su[i].parameters]
            bounds = self.bounds_su[:, :]

            assert params.size == 6
            assert bounds.shape == (6, 2)
        else:
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
            return self.Tdof2dof[:max(0, i)]

    def set_params_at(self, i, params):
        """
        Set DH parameters
        
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
            self.Tdof2dof[i-1].set_params(params[:4])
            self.Tdof2vdof[i].set_params(params[4:6])
            self.Tvdof2su[i].set_params(params[6:])

def tfquat_to_pyquat(q):
    return Quaternion(axis=q[:3], angle=q[3]) 

def quaternion_l2_distance(q1, q2):
    """
    sources: 
    - https://fgiesen.wordpress.com/2013/01/07/small-note-on-quaternion-distance-metrics/
    - http://kieranwynn.github.io/pyquaternion/#accessing-individual-elements
    """
    return 2*(1 - np.dot(q1.elements, q2.elements))

def quaternion_from_two_vectors(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    axis = np.cross(v1, v2)
    costh = np.dot(v1, v2)

    angle = np.arccos(costh)

    return Quaternion(axis=axis, angle=angle)