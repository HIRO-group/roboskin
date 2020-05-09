import sys
import numpy as np
import nlopt

# import robotic_skin
import robotic_skin.const as C
from robotic_skin.calibration.parameter_manager import get_IMU_pose
from robotic_skin.calibration.transformation_matrix import TransformationMatrix as TM


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
        Tdof2vdof = TM.from_numpy(params[:2], keys=['theta', 'd'])
        Tvdof2su = TM.from_numpy(params[2:])
    else:
        #     (4)    (2)     (4)
        # dof -> dof -> vdof -> su
        Tdof2vdof = TM.from_numpy(params[4:6], keys=['theta', 'd'])
        Tvdof2su = TM.from_numpy(params[6:])

    return Tdof2vdof * Tvdof2su


class Optimizer():
    """
    Optimizer class to evaluate the data.
    """
    def __init__(self, error_functions, stop_conditions=None, su_dhparams=None,
                 optimize_all=False):
        """
        Initializes the optimize with the following arguments:

        Arguments
        ---------
        `error_functions`:
            List of error functions to use during runtime

        `stop_condition`:
            When to stop optimizing the model

        `su_dhparams`:
            The DH parameters of the skin units on the robotic arm
        """
        self.error_functions = error_functions
        self.su_dhparams = su_dhparams
        self.optimize_all = optimize_all
        self.error_types = list(error_functions.keys())
        self.stop_conditions = stop_conditions
        self.target = self.error_types[0]
        self.all_poses = []
        if stop_conditions is None:
            self.stop_conditions = {'both': PassThroughStopCondition()}

    def optimize(self, i_imu, Tdofs, params, bounds):
        """
        Sets up the optimizer and runs the model.

        Arguments
        ---------
        `i_imu`
            Imu `i`

        `Tdofs`
            Transformation matrices from dof to dof

        `params`
            DH Parameters

        Returns
        -------
        `params`
            Predicted parameters from the model.
        """
        self.i_imu = i_imu
        self.Tdofs = Tdofs

        self.n_param = params.shape[0]
        # Construct an global optimizer
        opt = nlopt.opt(C.GLOBAL_OPTIMIZER, self.n_param)
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
        local_opt = nlopt.opt(C.LOCAL_OPTIMIZER, self.n_param)
        opt.set_local_optimizer(local_opt)

        # this is where most of the time is spent - in optimization
        params = opt.optimize(params)

        return params

    def objective(self, params, grad):
        """
        Objective function used in the optimizer.
        Called continuouly until a stopping condition is met.

        Arguments
        ---------
        `params`
            Predicted DH Parameters

        `grad`
            Gradient
        """
        Tdof2su = self.choose_true_or_estimated_Tdof2su(params)
        # self.Tdofs needs to be changed if we are optimizing all params.
        if self.optimize_all:
            modified_tdof = TM.from_numpy(params[:4])
            # update tdofs
            self.Tdofs[-1] = modified_tdof
        pos, quat = get_IMU_pose(self.Tdofs, Tdof2su)
        full_pose = np.r_[pos, quat]
        self.all_poses.append(full_pose)
        e = 0.0

        if sys.version_info[0] == 2:
            # iteritems() in python2
            for error_type, error_function in self.error_functions.iteritems():
                e += error_function(self.i_imu, self.Tdofs, Tdof2su)
        else:
            # items() in python3
            for error_type, error_function in self.error_functions.items():
                e += error_function(self.i_imu, self.Tdofs, Tdof2su)
        res = self.stop_conditions[self.target].update(params, None, e)
        return res

    def choose_true_or_estimated_Tdof2su(self, params):
        """
        Based on `self.dhparams`, determines
        if we want to use `convert_dhparams_to_Tdof2su`
        on `self.su_dhparams` or `params`.

        Arguments
        ---------
        `params`
            Currently estimated dh parameters.
        """
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
    Separate Optimizer class
    """
    def __init__(self, error_functions, stop_conditions, su_dhparams=None,
                 optimize_all=False):
        super().__init__(error_functions, su_dhparams=su_dhparams, optimize_all=optimize_all)
        """
        Initializes optimizer with selected error functions,
        certain stop conditions, and skin unit dh parameters.
        """
        # indices vary based on optimizing all dh params.
        if self.optimize_all:
            self.rotation_index = [0, 3, 4, 6, 9]
            self.position_index = [1, 2, 5, 7, 8]

        else:
            self.rotation_index = [0, 2, 5]
            self.position_index = [1, 3, 4]
        self.target = None
        self.error_functions = error_functions
        self.stop_conditions = stop_conditions

    def optimize(self, i_imu, Tdofs, params, bounds):
        """
        This function will optimize the given parameters in two steps.
        First it optimizes for rotational parameters and then
        optimizes for translational parameters. Hence, the name
        "Separate"Optimizer.

        i_imu: int
            ith IMU to be optimized

        Tdofs: list of TransformationMatrix
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
        self.params = params
        # Tdofs = self.param_manager.get_tmat_until(i_imu)

        n_param = params.shape[0]
        # optimizing half of them at each time
        self.n_param = int(n_param/2)

        # ################### First Optimize Rotations ####################
        # this takes care of 3 or 5 parameters at a time
        self.target = 'Rotation'
        self.stop_conditions['Rotation'].initialize()
        self.constant_params = params[self.position_index]

        opt = nlopt.opt(C.GLOBAL_OPTIMIZER, self.n_param)
        opt.set_min_objective(self.objective)
        opt.set_lower_bounds(bounds[self.rotation_index, 0])
        opt.set_upper_bounds(bounds[self.rotation_index, 1])
        opt.set_stopval(C.ROT_GLOBAL_STOP)
        local_opt = nlopt.opt(C.LOCAL_OPTIMIZER, self.n_param)
        opt.set_local_optimizer(local_opt)
        param_rot = opt.optimize(params[self.rotation_index])

        # ################### Then Optimize for Translations ####################
        self.target = 'Translation'
        self.stop_conditions['Translation'].initialize()
        self.constant_params = param_rot

        opt = nlopt.opt(C.GLOBAL_OPTIMIZER, self.n_param)
        opt.set_min_objective(self.objective)
        opt.set_lower_bounds(bounds[self.position_index, 0])
        opt.set_upper_bounds(bounds[self.position_index, 1])
        opt.set_stopval(C.POS_GLOBAL_STOP)
        local_opt = nlopt.opt(C.LOCAL_OPTIMIZER, self.n_param)
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
        Tdofs: list of TransformationMatrix
            Transformation Matrices between Dofs

        grad: np.ndarray
            Gradient, but we do not use any gradient information
            (We could in the future)
        Returns
        ----------
        error: float
            Error between measured values and estimated model outputs
        """
        # update self.Tdofs

        Tdof2su = self.choose_true_or_estimated_Tdof2su(
            np.r_[target_params, self.constant_params])
        # tdof is based on
        if self.optimize_all:
            first_params = self.current_params[:4]
            modified_tdof = TM.from_numpy(first_params)
            # update tdofs
            self.Tdofs[-1] = modified_tdof
        # target params from optimization thus far:
        # doesn't yet account for robot position, needed for later in 0's pose.
        pos, quat = get_IMU_pose(self.Tdofs, Tdof2su)
        full_pose = np.r_[pos, quat]
        self.all_poses.append(full_pose)
        # append pose
        e = self.error_functions[self.target](self.i_imu, self.Tdofs, Tdof2su)

        updated_params = self.stop_conditions[self.target].update(target_params, None, e)
        return updated_params

    def convert_dhparams_to_Tdof2su(self, merged_params):
        """
        converts current dh parameters to a
        transformation matrix from dof to skin unit.
        """
        # size depends on what we're optimizing.
        params = np.zeros(self.n_param * 2)
        if self.target == 'Rotation':

            params[self.rotation_index] = merged_params[:self.n_param]
            params[self.position_index] = merged_params[self.n_param:]
        elif self.target == 'Translation':
            params[self.rotation_index] = merged_params[self.n_param:]
            params[self.position_index] = merged_params[:self.n_param]
        self.current_params = params
        return convert_dhparams_to_Tdof2su(params)


class StopCondition():
    """
    Stop condition base class
    """
    def __init__(self):
        pass

    def initialize(self):
        pass

    def update(self, x, y, e):
        raise NotImplementedError()


class PassThroughStopCondition(StopCondition):
    """
    PassThroughStopCondition class.
    """
    def __init__(self):
        super().__init__()

    def update(self, x, y, e):
        return e


class DeltaXStopCondition(StopCondition):
    """
    DeltaXStopCondition class. Keeps track on the
    differences in x from iteration to iteration,
    until the updates are very small.
    """
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
