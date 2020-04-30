"""
The stats generator file will output the following stats:
1) A graph with SU's euclidean distance between real and predicted points.
    One legend will be Mittendorfer's method and another will be ours
2) A table comparing the dh params individually of our method and the others
3) A table comparing the orientation differences between Mittendorfer's method and ours
"""
from robotic_skin.calibration.optimizer import *
from robotic_skin.calibration.error_functions import *
from robotic_skin.calibration.loss import *
from robotic_skin.calibration.calibrate_imu_poses import load_data
from robotic_skin.calibration.calibrate_imu_poses import KinematicEstimator
from robotic_skin.calibration.calibrate_imu_poses import parse_arguments
from robotic_skin.calibration.utils import load_robot_configs
import matplotlib.pyplot as plt

if __name__ == "__main__":
    args = parse_arguments()
    measured_data = load_data(args.robot)
    robot_configs = load_robot_configs(args.configdir, args.robot)
    # Below code is for
    # 1) A graph with SU's euclidean distance between real and predicted points.
    #     One legend will be Mittendorfer's method and another will be ours

    # Method 1
    method1_name = "Our Method"
    error_functions = {
        'Rotation': StaticErrorFunction(measured_data, L2Loss()),
        'Translation': ConstantRotationErrorFunction(measured_data, L2Loss())
    }
    stop_conditions = {
        'Rotation': PassThroughStopCondition(),
        'Translation': DeltaXStopCondition()
    }
    optimizer_function = SeparateOptimizer
    method1_kinematics_estimator = KinematicEstimator(measured_data, robot_configs, optimizer_function,
                                                      error_functions, stop_conditions)
    method1_kinematics_estimator.optimize()

    # Method 2
    method2_name = "Modified Mittendorfer's Method"
    error_functions = {
        'Rotation': StaticErrorFunction(measured_data, L2Loss()),
        'Translation': MaxAccelerationErrorFunction(measured_data, L2Loss())
    }
    stop_conditions = {
        'Rotation': PassThroughStopCondition(),
        'Translation': DeltaXStopCondition()
    }
    optimizer_function = SeparateOptimizer
    method2_kinematics_estimator = KinematicEstimator(measured_data, robot_configs, optimizer_function,
                                                      error_functions, stop_conditions)
    method2_kinematics_estimator.optimize()

    # Method 2
    method3_name = "Mittendorfer's Method"
    error_functions = {
        'Rotation': StaticErrorFunction(measured_data, L2Loss()),
        'Translation': MaxAccelerationErrorFunction(measured_data, L2Loss(),
                                                    use_modified_mittendorfer=False)
    }
    stop_conditions = {
        'Rotation': PassThroughStopCondition(),
        'Translation': DeltaXStopCondition()
    }
    optimizer_function = SeparateOptimizer
    method3_kinematics_estimator = KinematicEstimator(measured_data, robot_configs, optimizer_function,
                                                      error_functions, stop_conditions)
    method3_kinematics_estimator.optimize()

    plt.plot(method1_kinematics_estimator.all_euclidean_distances, "-b", label=method1_name)
    plt.plot(method2_kinematics_estimator.all_euclidean_distances, "-r", label=method2_name)
    plt.plot(method3_kinematics_estimator.all_euclidean_distances, "-g", label=method3_name)

    plt.legend(loc="upper left")
    plt.show()

    # End
