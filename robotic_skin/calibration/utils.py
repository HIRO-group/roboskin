import numpy as np 

class TransMat():
    def __init__(self, params=None, n_params=4):
        """
        theta, d, a, alpha are all DH parameters 
        For DH Parameters, please refer to this video
        https://robotacademy.net.au/lesson/denavit-hartenberg-notation/
        """
        self.n_params = n_params
        self.params = params
        if params is None:
            # initialize randomly
            self.params = np.random.rand(n_params)
        
        self.mat = self.transformation_matrix(params) 

    def transformation_matrix(self, params):
        if params.shape[0] == 1:
            th = params
            d, a, al = 0
        if params.shape[0] == 2:
            th, d = params
            a, al = 0.0, 0.0
        else: 
            th, d, a, al = params

        return np.array([
            [np.cos(th), -np.sin(th)*np.cos(al),  np.sin(th)*np.sin(al), a*np.cos(th)],
            [np.sin(th),  np.cos(th)*np.cos(al), -np.cos(th)*np.sin(al), a*np.sin(th)],
            [0,           np.sin(al),             np.cos(al),            d],
            [0,           0,                      0,                     1]
        ])

    def __mul__(self, mat):
        """
        In our implementation, we use * for dot product
        """
        return np.dot(self.max, mat)

    def __rmul__(self, mat):
        """
        In our implementation, we use * for dot product
        """
        return np.dot(mat, self,mat)

    def T(self):
        return self.mat.T

    @property
    def R(self):
        return self.mat[:3, :3]

    @property
    def position(self):
        return self.mat[:3, 3]

    def set_params(self, params):
        self.params = params
        self.mat = self.transformation_matrix(params) 
