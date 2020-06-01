import logging
import numpy as np
import nlopt

# import robotic_skin
import robotic_skin.const as C
from robotic_skin.calibration.error_functions import (
    ErrorFunction,
    StaticErrorFunction,
    CombinedErrorFunction,
    MaxAccelerationErrorFunction,
    # ConstantRotationErrorFunction,
)
from robotic_skin.calibration.stop_conditions import (
    StopCondition,
    DeltaXStopCondition,
    PassThroughStopCondition
)
from robotic_skin.calibration.loss import L2Loss
from robotic_skin.calibration.utils.io import n2s


def choose_optimizer(args, kinematic_chain, evaluator, data_logger, optimize_all):
    if args.method == 'OM':
        optimizer = OurMethodOptimizer(
            kinematic_chain, evaluator, data_logger,
            optimize_all, args.error_functions, args.stop_conditions)
    elif args.method == 'MM':
        optimizer = MittendorferMethodOptimizer(
            kinematic_chain, evaluator, data_logger,
            optimize_all, args.error_functions, args.stop_conditions, method='mittendorfer')
    elif args.method == 'mMM':
        optimizer = MittendorferMethodOptimizer(
            kinematic_chain, evaluator, data_logger,
            optimize_all, args.error_functions, args.stop_conditions, method='modified_mittendorfer')
    else:
        raise ValueError(f'There is no such method name={args.method}')

    return optimizer


class OptimizerBase():
    """
    TODO: All Optimizer will inherit this class
    """
    def __init__(self, kinematic_chain, evaluator, data_logger, optimize_all):
        self.kinematic_chain = kinematic_chain
        self.evaluator = evaluator
        self.data_logger = data_logger
        self.optimize_all = optimize_all

    def optimize(self):
        """
        Optimize, evaluate and log
        """
        raise NotImplementedError()

    @property
    def result(self) -> dict:
        """
        Return dict
        - average_errors of all SU
            - euclidean_distance
            - quaternion_distance
        - best
            - errors for all SU
                - euclidean_distance
                - quaternion_distance
            - params for all SU
            - positions for all SU
            - orientations for all Su
        - trials
            - params
            - pose
            - positions
            - orientations
        """
        raise NotImplementedError()


class IncrementalOptimizerBase(OptimizerBase):
    def __init__(self, kinematic_chain, evaluator, data_logger, optimize_all,
                 error_functions, stop_conditions):
        super().__init__(kinematic_chain, evaluator, data_logger, optimize_all)
        if not (isinstance(error_functions, dict) or isinstance(error_functions, ErrorFunction)):
            raise ValueError('error_functions must be either dict or ErrorFunction')
        self.error_functions = error_functions
        self.stop_conditions = stop_conditions
        self.global_step = 0
        self.local_step = 0

    def optimize(self, data):
        """
        Optimize SU from Base to the End-Effector incrementally
        """
        self.data_logger.start_timer('total')
        # Initilialize error functions with data
        if isinstance(self.error_functions, dict):
            for error_function in self.error_functions.values():
                error_function.initialize(data)
        elif isinstance(self.error_functions, ErrorFunction):
            self.error_functions.initialize(data)

        logging.info('Skipping 0th IMU')
        for i_su in range(1, self.kinematic_chain.n_su):
            logging.info("Optimizing %ith SU ..." % (i_su))

            # optimize parameters wrt data
            self.data_logger.start_timer(timer_name=f'SU{i_su+1}')
            params = self._optimize(i_su=i_su)
            elapsed_time = self.data_logger.end_timer(timer_name=f'SU{i_su+1}')

            # Compute necessary data
            self.kinematic_chain.set_params_at(i_su, params)
            T = self.kinematic_chain.compute_su_TM(i_su, pose_type='eval')

            # Evalute and print to terminal
            errors = self.evaluator.evaluate(i_su=i_su, T=T)
            # Append to a logger and save every loop
            self.data_logger.add_best(
                i_su=i_su,
                params=params,
                position=T.position,
                orientation=T.quaternion,
                elapsed_time=elapsed_time,
                euclidean_distance=errors['position'],
                quaternion_distance=errors['orientation'])

            logging.info('='*100)
            logging.info(f'Position: {T.position}')
            logging.info(f'Quaternion: {T.quaternion}')
            logging.info(f"Euclidean distance: {errors['position']}")
            logging.info(f'Elapsed Time {elapsed_time}')
            logging.info('='*100)
        elapsed_time = self.data_logger.end_timer('total')

    def _optimize(self, i_su: int):
        """
        Optimize i_su sensor
        """
        # Initialize local_step when this function is called
        self.local_step = 0

        # Increement internally at every optimization step
        self.global_step += 1
        self.local_step += 1
        raise NotImplementedError


class NNOptimizerBase(OptimizerBase):
    def __init__(self, kinematic_chain, evaluator, data_logger, optimize_all):
        super().__init__(kinematic_chain, evaluator, data_logger)

    def optimize(self):
        """
        NN opimization
        """
        raise NotImplementedError()


class MixedIncrementalOptimizer(IncrementalOptimizerBase):
    def __init__(self, kinematic_chain, evaluator, data_logger, optimize_all,
                 error_function: ErrorFunction, stop_condition: StopCondition):
        super().__init__(kinematic_chain, evaluator, data_logger, optimize_all,
                         error_function, stop_condition)

    def _optimize(self, i_su):
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
        self.i_su = i_su
        self.local_step = 0

        self.n_param = params.shape[0]
        self.stop_conditions.initialize()
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

        params, _ = self.kinematic_chain.get_params_at(i_su=self.i_su)
        e = self.error_functions(self.kinematic_chain, self.i_su)
        res, params = self.stop_conditions.update(params, None, e)

        # Evaluate
        T = self.kinematic_chain.compute_su_TM(self.i_su, pose_type='eval')
        errors = self.evaluator.evaluate(i_su=self.i_su, T=T)
        # Log
        self.data_logger.add_trial(
            global_step=self.global_step,
            params=params,
            position=T.position,
            orientation=T.quaternion,
            euclidean_distance=errors['position'],
            quaternion_distance=errors['orientation'])
        # print to terminal
        logging.debug(f'e={e:.5f}, res={res:.5f}, params:{n2s(params, 3)} ' +
                      f'P:{n2s(T.position, 3)}, Q:{n2s(T.quaternion, 3)}')

        self.local_step += 1
        self.global_step += 1

        return res


class SeparateIncrementalOptimizer(IncrementalOptimizerBase):
    def __init__(self, kinematic_chain, evaluator, data_logger,
                 optimize_all, error_functions, stop_conditions):
        super().__init__(kinematic_chain, evaluator, data_logger, optimize_all,
                         error_functions, stop_conditions)
        self.targets = ['Position', 'Orientation']

        for dictionary in [error_functions, stop_conditions]:
            if self.targets != list(dictionary.keys()):
                raise KeyError(f'All dict Keys must be {self.targets}')

        self.indices = {}
        # indices vary based on optimizing all dh params.
        if self.optimize_all:
            self.indices['Orientation'] = [0, 3, 4, 6, 9]
            self.indices['Position'] = [1, 2, 5, 7, 8]
        else:
            self.indices['Orientation'] = [0, 2, 5]
            self.indices['Position'] = [1, 3, 4]

    def _optimize(self, i_su):
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

        # optimizing half of them at each time
        self.n_param = int(params.shape[0]/2)

        # ################### First Optimize Rotations ####################
        self.target = 'Orientation'
        self.constant = 'Position'
        self.target_index = self.indices[self.target]
        self.constant_index = self.indices[self.constant]
        self.stop_conditions[self.target].initialize()
        self.constant_params = params[self.constant_index]

        opt = nlopt.opt(C.GLOBAL_OPTIMIZER, self.n_param)
        opt.set_min_objective(self.objective)
        opt.set_lower_bounds(bounds[self.target_index, 0])
        opt.set_upper_bounds(bounds[self.target_index, 1])
        opt.set_stopval(C.ROT_GLOBAL_STOP)
        local_opt = nlopt.opt(C.LOCAL_OPTIMIZER, self.n_param)
        opt.set_local_optimizer(local_opt)
        params[self.target_index] = opt.optimize(params[self.target_index])

        # ################### Then Optimize for Translations ####################
        self.target = 'Position'
        self.constant = 'Orientation'
        self.target_index = self.indices[self.target]
        self.constant_index = self.indices[self.constant]
        self.stop_conditions[self.target].initialize()
        self.constant_params = params[self.constant_index]

        opt = nlopt.opt(C.GLOBAL_OPTIMIZER, self.n_param)
        opt.set_min_objective(self.objective)
        opt.set_lower_bounds(bounds[self.target_index, 0])
        opt.set_upper_bounds(bounds[self.target_index, 1])
        opt.set_stopval(C.POS_GLOBAL_STOP)
        local_opt = nlopt.opt(C.LOCAL_OPTIMIZER, self.n_param)
        opt.set_local_optimizer(local_opt)
        params[self.target_index] = opt.optimize(params[self.target_index])

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

        params, _ = self.kinematic_chain.get_params_at(i_su=self.i_su)
        e = self.error_functions[self.target](self.kinematic_chain, self.i_su)
        res, target_params = self.stop_conditions[self.target].update(params[self.target_index], None, e)

        # Evaluate
        T = self.kinematic_chain.compute_su_TM(self.i_su, pose_type='eval')
        errors = self.evaluator.evaluate(i_su=self.i_su, T=T)
        # Log
        self.data_logger.add_trial(
            global_step=self.global_step,
            params=params,
            position=T.position,
            orientation=T.quaternion,
            euclidean_distance=errors['position'],
            quaternion_distance=errors['orientation'])
        # print to terminal
        logging.debug(f'e={e:.5f}, res={res:.5f}, params:{n2s(target_params, 3)} ' +
                      f'P:{n2s(T.position, 3)}, Q:{n2s(T.quaternion, 3)}')

        self.local_step += 1
        self.global_step += 1

        return res


class MittendorferMethodOptimizer(MixedIncrementalOptimizer):
    def __init__(self, kinematic_chain, evaluator, data_logger,
                 optimize_all, error_function_=None, stop_condition_=None,
                 method='mittendorfer'):
        error_function = CombinedErrorFunction(
            e1=StaticErrorFunction(
                loss=L2Loss()),
            e2=MaxAccelerationErrorFunction(
                loss=L2Loss(),
                method=method)
        )
        stop_condition = DeltaXStopCondition()

        if isinstance(error_function_, ErrorFunction):
            error_function = error_function_
        if isinstance(stop_condition_, StopCondition):
            stop_condition = stop_condition_

        super().__init__(kinematic_chain, evaluator, data_logger,
                         optimize_all, error_function, stop_condition)


class OurMethodOptimizer(SeparateIncrementalOptimizer):
    def __init__(self, kinematic_chain, evaluator, data_logger, optimize_all,
                 error_functions_=None, stop_conditions_=None, method='our'):
        error_functions = {
            'Position': MaxAccelerationErrorFunction(L2Loss(), method=method),
            'Orientation': StaticErrorFunction(L2Loss())}
        stop_conditions = {
            'Position': DeltaXStopCondition(),
            'Orientation': PassThroughStopCondition(),
        }

        if isinstance(error_functions_, dict):
            for k, v in error_functions_.items():
                if k in error_functions.keys():
                    error_functions[k] = v
        if isinstance(stop_conditions_, dict):
            for k, v in stop_conditions_.items():
                if k in error_functions.keys():
                    stop_conditions[k] = v
                    print('Stop Condition Set')

        super().__init__(kinematic_chain, evaluator, data_logger,
                         optimize_all, error_functions, stop_conditions)
