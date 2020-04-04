import sys
import numpy as np
import nlopt

# import robotic_skin
import robotic_skin.const as C
from robotic_skin.calibration.utils import TransMat, get_IMU_pose


def convert_dhparams_to_Tdof2su(params):
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
    def __init__(self, error_functions, stop_conditions=None, su_dhparams=None):
        """
        """
        self.error_functions = error_functions
        self.su_dhparams = su_dhparams
        self.error_types = list(error_functions.keys())
        self.stop_conditions = stop_conditions
        self.target = self.error_types[0]
        if stop_conditions is None:
            self.stop_conditions = {'both': PassThroughStopCondition()}

    def optimize(self, i_imu, Tdofs, params, bounds):
        """
        """
        self.i_imu = i_imu
        self.Tdofs = Tdofs

        n_param = params.shape[0]
        # Construct an global optimizer
        opt = nlopt.opt(C.GLOBAL_OPTIMIZER, n_param)
        # The objective function only accepts x and grad arguments.
        # This is the only way to pass other arguments to opt
        # https://github.com/JuliaOpt/NLopt.jl/issues/27
        self.previous_params = None
        opt.set_min_objective(self.objective)
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

    def objective(self, params, grad):
        Tdof2su = self.choose_true_or_estimated_Tdof2su(params)

        pos, quat = get_IMU_pose(self.Tdofs, Tdof2su)

        e = 0.0
        if sys.version_info[0] == 2:
            for error_type, error_function in self.error_functions.iteritems():
                e += error_function(self.i_imu, self.Tdofs, Tdof2su)
        else:
            for error_type, error_function in self.error_functions.items():
                e += error_function(self.i_imu, self.Tdofs, Tdof2su)

        return self.stop_conditions[self.target].update(params, None, e)

    def choose_true_or_estimated_Tdof2su(self, params):
        if self.su_dhparams is not None:
            return convert_dhparams_to_Tdof2su(
                self.su_dhparams['su%i' % (self.i_imu+1)])
        else:
            return self.convert_dhparams_to_Tdof2su(params)

    def convert_dhparams_to_Tdof2su(self, params):
        """
        """
        return convert_dhparams_to_Tdof2su(params)


class SeparateOptimizer(Optimizer):
    """
    """
    def __init__(self, error_functions, stop_conditions, su_dhparams=None):
        super().__init__(error_functions, su_dhparams=su_dhparams)
        """
        """
        self.rotation_index = [0, 2, 5]
        self.position_index = [1, 3, 4]
        self.target = None
        self.error_functions = error_functions
        self.stop_conditions = stop_conditions

    def optimize(self, i_imu, Tdofs, params, bounds):
        """
        This function will optimize the given parameters in two steps.
        First it optimizes for rotational parameters and then
        optimizes for translational parameters.

        i_imu: int
            ith IMU to be optimized

        Tdofs: list of TransMat
            List of Transformation Matrices from 0th to ith Joint (Not to SU)

        params: list of float
            DH parameters to be optimized.
            This will be converted to Tdof2su which is a transformation
            matrix from ith Joint to ith SU (6 parameters)
            If the number of parameters were 10, it also includes
            4 DH parameter for i-1th to the ith Joint.

        bounds: np.ndarray
            Boundaries (Min and Max) for each parameter
        """
        self.i_imu = i_imu
        self.Tdofs = Tdofs

        n_param = params.shape[0]
        # optimizing half of them at each time
        n_param = int(n_param/2)

        # ################### First Optimize Rotations ####################
        self.target = 'Rotation'
        self.stop_conditions['Rotation'].initialize()
        self.constant_params = params[self.position_index]

        opt = nlopt.opt(C.GLOBAL_OPTIMIZER, n_param)
        opt.set_min_objective(self.objective)
        opt.set_lower_bounds(bounds[self.rotation_index, 0])
        opt.set_upper_bounds(bounds[self.rotation_index, 1])
        opt.set_stopval(C.ROT_GLOBAL_STOP)
        local_opt = nlopt.opt(C.LOCAL_OPTIMIZER, n_param)
        opt.set_local_optimizer(local_opt)
        param_rot = opt.optimize(params[self.rotation_index])

        # ################### Then Optimize for Translations ####################
        self.target = 'Translation'
        self.stop_conditions['Translation'].initialize()
        self.constant_params = param_rot

        opt = nlopt.opt(C.GLOBAL_OPTIMIZER, n_param)
        opt.set_min_objective(self.objective)
        opt.set_lower_bounds(bounds[self.position_index, 0])
        opt.set_upper_bounds(bounds[self.position_index, 1])
        opt.set_stopval(C.POS_GLOBAL_STOP)
        local_opt = nlopt.opt(C.LOCAL_OPTIMIZER, n_param)
        opt.set_local_optimizer(local_opt)
        param_pos = opt.optimize(params[self.position_index])

        params[self.rotation_index] = param_rot
        params[self.position_index] = param_pos

        return params

    def objective(self, target_params, grad):
        """
        Computes an error e_T = e_1 + e_2 from current parameters

        Arguments
        ----------
        target_params: list of floats
            Target parameters to be estimated

        constant_params: list of floats
            Parameters that are not to be optimized but used

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
        Tdof2su = self.choose_true_or_estimated_Tdof2su(
            np.r_[target_params, self.constant_params])

        pos, quat = get_IMU_pose(self.Tdofs, Tdof2su)

        e = self.error_functions[self.target](self.i_imu, self.Tdofs, Tdof2su)

        return self.stop_conditions[self.target].update(target_params, None, e)

    def convert_dhparams_to_Tdof2su(self, merged_params):
        """
        """
        params = np.zeros(6)

        if self.target == 'Rotation':
            params[self.rotation_index] = merged_params[:3]
            params[self.position_index] = merged_params[3:]
        elif self.target == 'Translation':
            params[self.rotation_index] = merged_params[3:]
            params[self.position_index] = merged_params[:3]

        return convert_dhparams_to_Tdof2su(params)


class StopCondition():
    def __init__(self):
        pass

    def initialize(self):
        pass

    def update(self, x, y, e):
        raise NotImplementedError()


class PassThroughStopCondition(StopCondition):
    def __init__(self):
        super().__init__()

    def update(self, x, y, e):
        return e


class DeltaXStopCondition(StopCondition):
    def __init__(self, windowsize=10, threshold=0.001, retval=0.00001):
        super().__init__()

        self.initialize()
        self.windowsize = windowsize
        self.threshold = threshold
        self.retval = retval

    def initialize(self):
        self.prev_x = None
        self.xdiff = None
        self.xdiffs = np.array([])

    def update(self, x, y, e):
        if self.prev_x is None:
            self.xdiff = None
            self.prev_x = np.array(x)
        else:
            self.xdiff = np.linalg.norm(np.array(x) - self.prev_x)
            self.prev_x = np.array(x)

        if self.xdiff is not None:
            self.xdiffs = np.append(self.xdiffs, self.xdiff)

        if len(self.xdiffs) >= self.windowsize:
            if np.mean(self.xdiffs[-(self.windowsize+1):-1]) <= self.threshold:
                return self.retval

        return e
