import numpy as np
import nlopt

# import robotic_skin
import robotic_skin.const as C
from robotic_skin.calibration.utils import TransMat

def convert_params_to_Tdof2su(params):
    """
    For the 1st IMU, we do not need to think of DoF-to-DoF transformation.
    Thus, 6 Params (2+4 for each IMU).
    Otherwise 10 = 4 (DoF to DoF) + 6 (IMU)
    This condition also checks whether DH params are passed or not
    If given, only the parameters for IMU should be estimated.
    """
    if params.shape[0] == 6:
        #     (2)     (4)
        # dof -> vdof -> su
        Tdof2vdof = TransMat(params[:2])
        Tvdof2su = TransMat(params[2:])
    else:
        #     (4)    (2)     (4)
        # dof -> dof -> vdof -> su
        Tdof2vdof = TransMat(params[4:6])
        Tvdof2su = TransMat(params[6:])

    return Tdof2vdof.dot(Tvdof2su)

class Optimizer():
    """
    """
    def __init__(self, error_function):
        """
        """
        self.error_function = error_function
        self.previous_params = None

    def optimize(self, i, params, bounds, Tdofs):
        """
        """
        n_param = params.shape[0]
        # Construct an global optimizer
        opt = nlopt.opt(C.GLOBAL_OPTIMIZER, n_param)
        # The objective function only accepts x and grad arguments.
        # This is the only way to pass other arguments to opt
        # https://github.com/JuliaOpt/NLopt.jl/issues/27
        self.previous_params = None
        opt.set_min_objective(lambda x, grad: self.error_function(x, grad, i, Tdofs, params_to_Tdof2su))
        # Set boundaries
        opt.set_lower_bounds(bounds[:, 0])
        opt.set_upper_bounds(bounds[:, 1])
        # set stopping threshold
        opt.set_stopval(C.GLOBAL_STOP)
        # Need to set a local optimizer for the global optimizer
        local_opt = nlopt.opt(C.LOCAL_OPTIMIZER, n_param)
        opt.set_local_optimizer(local_opt)

        # this is where most of the time is spent - in optimization
        params = opt.optimize(params)

        return params

    def params_to_Tdof2su(self, params):
        """
        """
        return convert_params_to_Tdof2su(params)

class SeperateOptimizer(Optimizer):
    """
    """
    def __init__(self, error_function):
        super().__init__(error_function)
        """
        """
        self.rot_index = [0, 2, 5]
        self.pos_index = [1, 3, 4]
        self.previous_params = None
        self.parameter_diffs = np.array([])

    def optimize(self, i, params, bounds, Tdofs):
        """
        """
        n_param = params.shape[0]
        # ################### First Optimize Rotations ####################
        n_param = int(n_param/2)
        param_rot = params[self.rot_index]
        param_pos = params[self.pos_index]

        opt = nlopt.opt(C.GLOBAL_OPTIMIZER, n_param)
        opt.set_min_objective(lambda x, grad: self.error_function(x, grad, i, Tdofs, param_pos, 'rot', self.params_to_Tdof2su))
        opt.set_lower_bounds(bounds[self.rot_index, 0])
        opt.set_upper_bounds(bounds[self.rot_index, 1])
        opt.set_stopval(C.ROT_GLOBAL_STOP)
        local_opt = nlopt.opt(C.LOCAL_OPTIMIZER, n_param)
        opt.set_local_optimizer(local_opt)
        param_rot = opt.optimize(param_rot)
        print(param_rot)

        # ################### Then Optimize for Translations ####################
        self.parameter_diffs = np.array([])
        opt = nlopt.opt(C.GLOBAL_OPTIMIZER, n_param)
        opt.set_min_objective(lambda x, grad: self.error_function(x, grad, i, Tdofs, param_rot, 'pos', self.params_to_Tdof2su))
        opt.set_lower_bounds(bounds[self.pos_index, 0])
        opt.set_upper_bounds(bounds[self.pos_index, 1])
        opt.set_stopval(C.POS_GLOBAL_STOP)
        local_opt = nlopt.opt(C.LOCAL_OPTIMIZER, n_param)
        opt.set_local_optimizer(local_opt)
        param_pos = opt.optimize(param_pos)
        print(param_pos)

        params[self.rot_index] = param_rot
        params[self.pos_index] = param_pos

        return params

    def params_to_Tdof2su(self, target_params, const_params):
        """
        """
        params = np.zeros(6)

        if target == 'rot':
            params[self.rot_index] = target_params
            params[self.pos_index] = const_params
        elif target == 'pos':
            params[self.rot_index] = const_params
            params[self.pos_index] = target_params

        return convert_params_to_Tdof2su(params)


    def error_function(self, target_params, grad, i, Tdofs, const_params, target):
        """
        Computes an error e_T = e_1 + e_2 from current parameters

        Arguments
        ----------
        params: np.ndarray
            Current estimated parameters

        i: int
            ith sensor
        Tdofs: list of TransMat
            Transformation Matrices between Dofs

        grad: np.ndarray
            Gradient, but we do not use any gradient information
            (We could in the future)
        Returns
        ----------
        error: float
            Error between measured values and estimated model outputs
        """
        Tdof2su = self.optimizer.params_to_Tdof2su(target_params, const_params)
        # Tdof2su = convert_params_to_Tdof2su(self.robot_configs['su_dh_parameter']['su%i' % (i+1)])

        pos, quat = get_IMU_pose(Tdofs, Tdof2su)

        if self.previous_params is None:
            self.xdiff = None
            self.previous_params = np.array(target_params)
        else:
            self.xdiff = np.linalg.norm(np.array(target_params) - self.previous_params)
            self.previous_params = np.array(target_params)

        if target == 'rot':
            e1 = self.static_error_function(i, Tdofs, Tdof2su)
            print('IMU'+str(i), n2s(e1, 5), n2s(params), n2s(pos), n2s(quat))
            # e4 = np.sum(np.abs(params)[[0,2,5]])
            return e1
        else:
            # e2 = self.dynamic_error_function(i, Tdofs, Tdof2su)
            # print(n2s(e2, 5), n2s(params), n2s(pos), n2s(quat))
            # return e2
            e3 = self.rotation_error_function(i, Tdofs, Tdof2su)
            print('IMU'+str(i), n2s(e3, 5), n2s(params), n2s(pos), n2s(quat), self.xdiff)

            if len(self.parameter_diffs) >= 10:
                if np.mean(self.parameter_diffs[-11:-1]) <= 0.001:
                    return 0.00001

            return e3
