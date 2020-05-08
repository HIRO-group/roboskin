import numpy as np
from .tmatrix import TransformationMatrix as TM


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
            self.Tdof2dof = [TM.from_bounds(bounds=bounds) for i in range(n_joint)]
        else:
            # 6 parameters to optimize.
            self.Tdof2dof = [TM.from_list(dhparams['joint' + str(i+1)]) for i in range(n_joint)]

        self.Tdof2vdof = [TM.from_bounds(
            bounds=bounds_su[:2, :],
            keys=['theta', 'd']) for i in range(n_joint)]
        self.Tvdof2su = [TM.from_bounds(bounds=bounds_su[2:, :]) for i in range(n_joint)]

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
        list of TransformationMatrix
            Transformation Matrices between DoFs
        list of TransformationMatrix
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
            self.Tdof2vdof[i].set_params(params[:2], keys=['theta', 'd'])
            self.Tvdof2su[i].set_params(params[2:])
        else:
            self.Tdof2dof[i].set_params(params[:4])
            self.Tdof2vdof[i].set_params(params[4:6], keys=['theta', 'd'])
            self.Tvdof2su[i].set_params(params[6:])
