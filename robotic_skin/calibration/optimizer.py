import logging
import numpy as np
import nlopt

# import robotic_skin
import robotic_skin.const as C
from robotic_skin.calibration.stop_conditions import PassThroughStopCondition
from robotic_skin.calibration.utils.io import n2s


class Optimizer():
    """
    Optimizer class to evaluate the data.
    """
    def __init__(self, kinematic_chain, error_functions,
                 stop_conditions=None, optimize_all=False):
        """
        Initializes the optimize with the following arguments:

        Arguments
        ---------
        error_functions:
            List of error functions to use during runtime

        stop_condition:
            When to stop optimizing the model

        su_dhparams:
            The DH parameters of the skin units on the robotic arm
        """
        self.kinematic_chain = kinematic_chain
        self.error_functions = error_functions
        self.optimize_all = optimize_all
        self.error_types = list(error_functions.keys())
        self.stop_conditions = stop_conditions
        self.target = self.error_types[0]
        self.all_poses = []
        if stop_conditions is None:
            self.stop_conditions = {'both': PassThroughStopCondition()}

    def optimize(self, i_su):
        """
        Sets up the optimizer and runs the model.

        Arguments
        ---------
        i_su: int
            i_su th Su

        Returns
        -------
        params
            Predicted parameters from the model.
        """
        params, bounds = self.kinematic_chain.get_params_at(i_su=i_su)

        self.all_poses = []
        self.i_su = i_su

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
        self.kinematic_chain.set_params_at(self.i_su, params)
        T = self.kinematic_chain.compute_su_TM(self.i_su, pose_type='eval')

        self.all_poses.append(np.r_[T.position, T.quaternion])
        e = 0.0

        params, _ = self.kinematic_chain.get_params_at(self.i_su)
        for _, error_function in self.error_functions.items():
            e += error_function(self.kinematic_chain, self.i_su)
        res = self.stop_conditions[self.target].update(params, None, e)

        logging.info(f'e={res}, P:{T.position}, Q:{T.quaternion}')  # noqa: E999
        return res


class SeparateOptimizer(Optimizer):
    """
    Separate Optimizer class
    """
    def __init__(self, kinematic_chain, error_functions,
                 stop_conditions=None, optimize_all=False):
        super().__init__(kinematic_chain, error_functions,
                         optimize_all=optimize_all)
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

    def optimize(self, i_su):
        """
        This function will optimize the given parameters in two steps.
        First it optimizes for rotational parameters and then
        optimizes for translational parameters. Hence, the name
        "Separate"Optimizer.

        Arguments
        ---------
        i_su: int
            ith IMU to be optimized
        """
        params, bounds = self.kinematic_chain.get_params_at(i_su=i_su)

        self.i_su = i_su
        self.all_poses = []

        n_param = params.shape[0]
        # optimizing half of them at each time
        self.n_param = int(n_param/2)

        # ################### First Optimize Rotations ####################
        logging.info('Optimizing Rotation')
        self.target = 'Rotation'
        self.target_index = self.rotation_index
        self.constant_index = self.position_index
        self.stop_conditions['Rotation'].initialize()
        self.constant_params = params[self.constant_index]

        opt = nlopt.opt(C.GLOBAL_OPTIMIZER, self.n_param)
        opt.set_min_objective(self.objective)
        opt.set_lower_bounds(bounds[self.rotation_index, 0])
        opt.set_upper_bounds(bounds[self.rotation_index, 1])
        opt.set_stopval(C.ROT_GLOBAL_STOP)
        local_opt = nlopt.opt(C.LOCAL_OPTIMIZER, self.n_param)
        opt.set_local_optimizer(local_opt)
        param_rot = opt.optimize(params[self.rotation_index])

        # ################### Then Optimize for Translations ####################
        logging.info('Optimizing Translation')
        self.target = 'Translation'
        self.target_index = self.position_index
        self.constant_index = self.rotation_index
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
        grad: np.ndarray
            Gradient, but we do not use any gradient information
            (We could in the future)

        Returns
        ----------
        error: float
            Error between measured values and estimated model outputs
        """
        # update self.Tdofs
        params = np.zeros(self.n_param * 2)
        params[self.target_index] = target_params
        params[self.constant_index] = self.constant_params

        self.kinematic_chain.set_params_at(self.i_su, params)
        T = self.kinematic_chain.compute_su_TM(self.i_su, pose_type='eval')

        # append pose
        self.all_poses.append(np.r_[T.position, T.quaternion])

        params, _ = self.kinematic_chain.get_params_at(i_su=self.i_su)
        e = self.error_functions[self.target](self.kinematic_chain, self.i_su)
        # if self.target == 'Rotation':
        #     e += np.linalg.norm(params[self.target_index])
        res = self.stop_conditions[self.target].update(params[self.target_index], None, e)

        logging.info(f'e={res}, {n2s(params, 3)}, P:{n2s(T.position, 3)}, Q:{n2s(T.quaternion, 3)}')
        return res
