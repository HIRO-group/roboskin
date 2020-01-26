import numpy as np 

class TransMat():
    """
    Class for Transformation Matrix
    Manages all its parameters and computation
    """
    def __init__(self, params=None):
        """
        theta, d, a, alpha are all DH parameters 
        For DH Parameters, please refer to this video
        https://robotacademy.net.au/lesson/denavit-hartenberg-notation/
        """
        self.params = params
        if params is None:
            # initialize randomly
            self.params = np.random.rand(4)
        self.n_params = self.params.size
        
        th, d, a, al = self.check_params(self.params)
        self.mat = self.transformation_matrix(th, d, a, al)

    def check_params(self, params):
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
        th:
            Rotation theta around z axis (rad)
        d: 
            Displacement relative to z axis (m)
        a:
            Displacement relative to x axis (m)
        al:
            Rotation alpha around x axis (rad) 
        """
        return np.array([
            [np.cos(th), -np.sin(th)*np.cos(al),  np.sin(th)*np.sin(al), a*np.cos(th)],
            [np.sin(th),  np.cos(th)*np.cos(al), -np.cos(th)*np.sin(al), a*np.sin(th)],
            [0,           np.sin(al),             np.cos(al),            d],
            [0,           0,                      0,                     1]
        ])

    def dhparameters(self, mat):
        th = np.arctan2(mat[1, 0], mat[0, 0])
        d = mat[2, 3]
        a = np.sqrt(np.square(mat[0, 3]) + np.square(mat[1, 3]))
        al = np.arctan2(mat[2, 1], mat[2, 2])
        return np.array([th, d, a, al])

    def __mul__(self, T):
        """
        In our implementation, we use * for dot product

        mat: np.ndarray
            4 by 4 ndarray tranformation matrix
        """
        new_mat = np.dot(T.mat, self.mat)
        params = self.dhparameters(new_mat)
        T = TransMat(params)
        T.mat = new_mat
        return T

    def __rmul__(self, T):
        """
        In our implementation, we use * for dot product
        
        mat: np.ndarray
            4 by 4 ndarray tranformation matrix
        """
        new_mat = np.dot(self.mat, T.mat)
        params = self.dhparameters(new_mat)
        T = TransMat(params)
        T.mat = new_mat
        return T

    def T(self):
        """
        Transpose
        """
        return self.mat.T

    @property
    def R(self):
        """
        Rotation Matrix
        """
        return self.mat[:3, :3]

    @property
    def position(self):
        """
        Position as a result of the tranformation
        """
        return self.mat[:3, 3]

    def set_params(self, params):
        """
        Set parameters that have been optimized

        params: np.array
            DH parameters
        """
        self.params = params
        th, d, a, al = self.check_params(self.params)
        self.mat = self.transformation_matrix(th, d, a, al)

    @property
    def parameters(self):
        if self.n_params == 1:
            return self.params[0]
        if self.n_params == 2:
            return self.params[[0, 1]]
        if self.n_params == 4:
            return self.params