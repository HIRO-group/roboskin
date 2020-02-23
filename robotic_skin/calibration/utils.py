"""
Utilities module for Robotic Skin.
"""
import numpy as np 

class TransMat():
    """
    Class for Transformation Matrix
    Manages all its parameters and computation
    It also outputs Rotation Matrix and Position of the transformed result
    """
    def __init__(self, params=None):
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
            # initialize randomly
            params = np.random.rand(4)
        self.n_params = params.size
        
        th, d, a, al = self.check_params(params)
        self.params = np.array([th, d, a, al]) 
        self.mat = self.transformation_matrix(th, d, a, al)

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

        The (n-1)th coordinate frame is 
        1. Displace for d [m] along z axis. 
        2. Rotate for th [rad] around z axis.
        3. Displace for a [m] along x axis.
        4. Rotate for al [rad] around x axis.
        to get the nth coordinate frame.

        ..math:: {}^{n-1}_{n}T = Tz * Rz * Tx * Rx
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
                {}^{SU_6}_{0}R = ({}^{0}_{SU_6}T).R_inv

            R_inv is implemented as a function in TransMat

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
        return np.array([
            [np.cos(th), -np.sin(th)*np.cos(al), np.sin(th)*np.sin(al), a*np.cos(th)],
            [np.sin(th), np.cos(th)*np.cos(al), -np.cos(th)*np.sin(al), a*np.sin(th)],
            [0, np.sin(al), np.cos(al), d],
            [0, 0, 0, 1]
        ])

    def dhparameters(self, mat):
        """
        Compute DH parameters from a transformation matrix

        Parameters
        -----------
        mat: np.ndarray
            A transformation matrix

        Returns
        -----------
        np.array
            DH Parameters
        """
        th = np.arctan2(mat[1, 0], mat[0, 0])
        d = mat[2, 3]
        a = np.sqrt(np.square(mat[0, 3]) + np.square(mat[1, 3]))
        al = np.arctan2(mat[2, 1], mat[2, 2])
        return np.array([th, d, a, al])

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
        params = self.dhparameters(new_mat)
        T = TransMat(params)
        T.mat = new_mat
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
        if self.n_params == 1:
            return self.params[0]
        elif self.n_params == 2:
            return self.params[[0, 1]]
        else:
            # self.n_params was specified as 4 or some other value,
            # return all of the values
            return self.params

class ParameterManager():
    """
    Class for managing DH parameters
    """
    def __init__(self, n_joint, poses, bounds, dhparams=None):
        """
        TODO For now, we assume n_sensor is equal to n_joint
        
        Arguments 
        -----------
        n_joints: int
            Number of joints
        poses: np.ndarray
            Poses for n_joints
        bounds: np.ndarray
            Bounds for DH parameters
        """
        self.n_joint = n_joint
        self.poses = poses
        self.bounds = bounds
        self.dhparams = dhparams 

        # TODO initialize with randomized value within a certain range
        if self.dhparams:
            self.Tdo2fdof = [TransMat(dhparams[i,:]) for i in range(n_joint-1)]
        else:
            self.Tdof2dof = [TransMat() for i in range(n_joint-1)]
        self.Tdof2vdof = [TransMat() for i in range(n_joint)]
        self.Tvdof2su = [TransMat(np.random.rand(2)) for i in range(n_joint)]
        self.Tposes = [[TransMat(np.array(theta)) for theta in pose] for pose in poses]

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
        if i == 0 or self.dhparams is not None:
            params = np.r_[self.Tdof2vdof[i].parameters, self.Tvdof2su[i].parameters]
            bounds = np.hstack([self.bounds[:, :], self.bounds[:, :2]])
        else:
            params = np.r_[self.Tdof2dof[i-1].parameters, 
                           self.Tdof2vdof[i].parameters, 
                           self.Tvdof2su[i].parameters]
            bounds = np.hstack([self.bounds[:, :], self.bounds[:, :], self.bounds[:, :2]])

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
            Tdof2dofs_until_i_joint = self.Tdof2dof[:i]
        else: 
            Tdof2dofs_until_i_joint = self.Tdof2dof[:max(0,i-1)]
        
        return Tdof2dofs_until_i_joint, [self.Tposes[p][:i+1] for p in range(self.poses.shape[0])]

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
        if i == 0 or self.dhparams is not None:
            self.Tdof2vdof[i].set_params(params[:4])
            self.Tvdof2su[i].set_params(params[4:])
        else:
            self.Tdof2dof[i-1].set_params(params[:4])
            self.Tdof2vdof[i].set_params(params[4:8])
            self.Tvdof2su[i].set_params(params[8:])