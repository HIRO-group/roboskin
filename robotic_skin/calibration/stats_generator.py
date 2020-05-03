"""
The stats generator file will output the following stats:
1) A graph with SU's euclidean distance between real and predicted points.
    One legend will be Mittendorfer's method and another will be ours
2) A table comparing the dh params individually of our method and the others
3) A table comparing the orientation differences between Mittendorfer's method and ours
"""
from robotic_skin.calibration.optimizer import (
    Optimizer,
    SeparateOptimizer,
    PassThroughStopCondition,
    DeltaXStopCondition
)
from robotic_skin.calibration.error_functions import (
    StaticErrorFunction,
    ConstantRotationErrorFunction,
    MaxAccelerationErrorFunction
)
from robotic_skin.calibration.loss import L2Loss
from robotic_skin.calibration.calibrate_imu_poses import load_data
from robotic_skin.calibration.calibrate_imu_poses import KinematicEstimator
from robotic_skin.calibration.calibrate_imu_poses import parse_arguments
from robotic_skin.calibration.utils import load_robot_configs
import matplotlib.pyplot as plt
import os
from tabulate import tabulate
import numpy as np


def array_to_table_string(dh_params_array: np.ndarray) -> list:
    return_list = []
    # Round off DH params to two decimals
    dh_params_array = np.around(dh_params_array, decimals=2)
    for each_accel_dhparams in dh_params_array:
        return_list.append(' , '.join([str(each_param) for each_param in each_accel_dhparams]))
    return return_list


def list_to_html_table(my_list: list, is_header: bool = False) -> str:
    starting_string = "<table><tr>"
    trailing_string = "</tr></table>"
    return_string = f"{starting_string}"
    for each_element in my_list:
        each_element = str(each_element)
        if is_header:
            return_string += f"<th>{each_element}</th>"
        else:
            return_string += f"<td>{each_element}</td>"
    return_string += trailing_string
    return return_string


class original_parameters:
    def __init__(self):
        repodir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config = os.path.join(repodir, 'config')
        robot_configs = load_robot_configs(config, 'panda')
        dh_params_array = []
        for each_su, dh_params in robot_configs['su_dh_parameter'].items():
            if each_su == 'su1':
                continue
            dh_params_array.append(dh_params)
        self.estimated_dh_params = dh_params_array
        orientation_array = []
        for each_su, su_contents in robot_configs['su_pose'].items():
            if each_su == 'su1':
                continue
            orientation_array.append(su_contents['rotation'])
        self.all_orientations = orientation_array


if __name__ == "__main__":
    args = parse_arguments()
    measured_data = load_data(args.robot)
    repodir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    configdir = os.path.join(repodir, 'config')
    robot_configs = load_robot_configs(configdir, args.robot)
    optimize_all = args.optimizeall

    stats_file_name = "stats.md"
    # Original_Params
    original_params = "OG"
    OG = original_parameters()

    # Method 1
    method1_name = "OM"
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
                                                      error_functions, stop_conditions, optimize_all)
    method1_kinematics_estimator.optimize()

    # Method 2
    method2_name = "MMM"
    error_functions = {
        'Rotation': StaticErrorFunction(measured_data, L2Loss()),
        'Translation': MaxAccelerationErrorFunction(measured_data, L2Loss())
    }
    stop_conditions = {
        'Rotation': DeltaXStopCondition(),
        'Translation': DeltaXStopCondition()
    }
    # optimization for each loss function is not done separately in mittendorfer's method

    optimizer_function = Optimizer
    method2_kinematics_estimator = KinematicEstimator(measured_data, robot_configs, optimizer_function,
                                                      error_functions, stop_conditions, optimize_all)
    method2_kinematics_estimator.optimize()

    # Method 3
    method3_name = "MM"
    error_functions = {
        'Rotation': StaticErrorFunction(measured_data, L2Loss()),
        'Translation': MaxAccelerationErrorFunction(measured_data, L2Loss(),
                                                    use_modified_mittendorfer=False)
    }
    stop_conditions = {
        'Rotation': DeltaXStopCondition(),
        'Translation': DeltaXStopCondition()
    }
    # optimization for each loss function is not done separately.

    optimizer_function = Optimizer
    method3_kinematics_estimator = KinematicEstimator(measured_data, robot_configs, optimizer_function,
                                                      error_functions, stop_conditions, optimize_all)
    method3_kinematics_estimator.optimize()

    # Below code is for
    # 1) A graph with SU's euclidean distance between real and predicted points.
    #     One legend will be Mittendorfer's method and another will be ours
    plt.plot(method1_kinematics_estimator.all_euclidean_distances, "-b", label=method1_name)
    plt.plot(method2_kinematics_estimator.all_euclidean_distances, "-r", label=method2_name)
    plt.plot(method3_kinematics_estimator.all_euclidean_distances, "-g", label=method3_name)
    plt.legend(loc="upper left")
    plt.show()

    open(stats_file_name, 'w').close()
    all_methods = [original_params, method1_name, method2_name, method3_name]
    all_kinematics_estimators = [OG, method1_kinematics_estimator, method2_kinematics_estimator,
                                 method3_kinematics_estimator]
    number_of_imus = method1_kinematics_estimator.n_sensor - 1
    # Below code is for
    # 1) A table comparing the dh params individually of our method and the others
    dh_parameter_headers = ["DH Parameters", "IMU 1", "IMU 2", "IMU 3", "IMU 4", "IMU 5", "IMU 6"]
    method_header = [""]
    for each_table_header in dh_parameter_headers[1:]:
        method_header.append(list_to_html_table(all_methods, True))
    table = []
    table.append(method_header)
    table_rows = ["Θ<sub>0</sub>", "d", "a", "α", "Θ<sub>acc</sub>", "d<sub>acc</sub>"]
    for i, each_table_row in enumerate(table_rows):
        individual_row = [each_table_row]
        for j in range(number_of_imus):
            individual_row.append(list_to_html_table([round(each_ke.estimated_dh_params[j][i], 2)
                                                      for each_ke in all_kinematics_estimators]))
        table.append(individual_row)
    print(tabulate(table, dh_parameter_headers, tablefmt="github"))
    with open("stats.md", "a") as f:
        f.write(tabulate(table, dh_parameter_headers, tablefmt="github").__str__())
        f.write("\n")
        f.write("\n")

    # Below code is for
    # 1) A table comparing the orientations individually of our method and the others
    orientation_headers = ["Orientations", "IMU 1", "IMU 2", "IMU 3", "IMU 4", "IMU 5", "IMU 6"]
    method_header = [""]
    for each_table_header in orientation_headers[1:]:
        method_header.append(list_to_html_table(all_methods, True))
    table = []
    table.append(method_header)
    table_rows = ["w", "x", "y", "z"]
    for i, each_table_row in enumerate(table_rows):
        individual_row = [each_table_row]
        for j in range(number_of_imus):
            individual_row.append(list_to_html_table([round(each_ke.all_orientations[j][i], 2)
                                                      for each_ke in all_kinematics_estimators]))
        table.append(individual_row)
    print(tabulate(table, orientation_headers, tablefmt="github"))
    with open(stats_file_name, "a") as f:
        f.write(tabulate(table, orientation_headers, tablefmt="github").__str__())
        f.write("\n")
        f.write("\n")
