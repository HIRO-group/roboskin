"""
The stats generator file will output the following stats:
1) A graph with SU's euclidean distance between real and predicted points.
    One legend will be Mittendorfer's method and another will be ours
2) A table comparing the dh params individually of our method and the others
3) A table comparing the orientation differences between Mittendorfer's method and ours
"""
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

from robotic_skin.calibration.optimizer import (
    OurMethodOptimizer,
    MittendorferMethodOptimizer,
)
from calibrate_imu_poses import (
    parse_arguments,
    construct_kinematic_chain,
    Evaluator,
    DataLogger
)
from robotic_skin.calibration import utils

REPODIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIGDIR = os.path.join(REPODIR, 'config')


def array_to_table_string(dh_params_array: np.ndarray) -> list:  # noqa: E999
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


def append_true_parameters(data_logger, robot_configs):
    su_dh_params = list(robot_configs['su_dh_parameter'].values())
    su_poses = list(robot_configs['su_pose'].values())

    for i_su, (params, su_pose) in enumerate(zip(su_dh_params, su_poses)):
        if i_su == 0:
            continue
        data_logger.add_best(
            i_su=i_su,
            params=params,
            orientation=su_pose['rotation'],
            euclidean_distance=0.0)


def add_image_to_md(filename, img_filename):
    open(filename, 'w').close()
    # Add the graph to readme
    with open(filename, 'a')as f:
        f.write(f"![]({img_filename})")
        f.write("\n")
        f.write("\n")


def save_plt(filename):
    if os.path.isfile(filename):
        # Matplotlib doesn't overwrite if file exists
        os.remove(filename)
    plt.savefig(filename, format="png")


def create_cascaded_table(method_names: list, data_loggers: list, target_measure: str,
                          column_names: list, row_names: list):
    if len(method_names) != len(data_loggers):
        raise ValueError('Size of method_names and data_loggers should be the same')

    table = []
    # SubHeader Row
    method_header_row = [""]
    for _ in range(len(column_names)):
        method_header_per_column = list_to_html_table(method_names, is_header=True)
        method_header_row.append(method_header_per_column)
    table.append(method_header_row)

    n_row = len(row_names)
    # Create the actual table
    # Create A Row
    for i, row_name in enumerate(row_names):
        # 1st Column
        individual_row = [row_name]
        # Loop to create each column
        for j in range(len(column_names)):
            # In each column, create n_methods cascaded columns
            columns = []
            # For n_methods data_loggers
            for data_logger in data_loggers:
                data = getattr(data_logger, target_measure)
                value = data[j] if data.ndim == 1 else data[j][i]
                columns.append(round(value, 2))
            individual_row.append(list_to_html_table(columns))
        table.append(individual_row)
    return table


def add_table_to_md(filename: str, table: list, headers: list):
    with open(filename, "a") as f:
        f.write(tabulate(table, headers, tablefmt="github").__str__())
        f.write("\n")
        f.write("\n")


if __name__ == "__main__":  # noqa: C901
    args = parse_arguments()
    datadir = utils.parse_datadir(args.datadir)
    measured_data, imu_mappings = utils.load_data(args.robot, datadir)

    robot_configs = utils.load_robot_configs(CONFIGDIR, args.robot)

    stats_file_name = os.path.join(REPODIR, "comparison_result.md")
    l2_norm_plot_file_name = os.path.join(REPODIR, "data", "l2_norm_graph2.png")

    # Original_Params
    data_logger_true = DataLogger(datadir, args.robot, args.method)
    append_true_parameters(data_logger_true, robot_configs)

    evaluator = Evaluator(true_su_pose=robot_configs['su_pose'])

    if args.test:
        kinematic_chain = construct_kinematic_chain(
            robot_configs, imu_mappings, args.test, args.optimizeall)
        data_logger1 = copy.deepcopy(data_logger_true)
        data_logger2 = copy.deepcopy(data_logger_true)
        data_logger3 = copy.deepcopy(data_logger_true)
    else:
        # Method 1
        kinematic_chain = construct_kinematic_chain(
            robot_configs, imu_mappings, args.test, args.optimizeall)
        data_logger1 = DataLogger(datadir, args.robot, args.method)
        optimizer1 = OurMethodOptimizer(
            kinematic_chain, evaluator, data_logger1,
            args.optimizeall, args.error_functions, args.stop_conditions)
        optimizer1.optimize(measured_data)

        # Method 2
        kinematic_chain = construct_kinematic_chain(
            robot_configs, imu_mappings, args.test, args.optimizeall)
        data_logger2 = DataLogger(datadir, args.robot, args.method)
        optimizer2 = MittendorferMethodOptimizer(
            kinematic_chain, evaluator, data_logger2,
            args.optimizeall, args.error_functions, args.stop_conditions, apply_normal_mittendorfer=True)
        optimizer2.optimize(measured_data)

        # Method 3
        kinematic_chain = construct_kinematic_chain(
            robot_configs, imu_mappings, args.test, args.optimizeall)
        data_logger3 = DataLogger(datadir, args.robot, args.method)
        optimizer3 = MittendorferMethodOptimizer(
            kinematic_chain, evaluator, data_logger3,
            args.optimizeall, args.error_functions, args.stop_conditions, apply_normal_mittendorfer=False)
        optimizer3.optimize(measured_data)

    n_su = kinematic_chain.n_su - 1  # Assume that the 1st IMU is skipped
    method_names = ['True', 'OM', "MM", "mMM"]
    data_loggers = [data_logger_true, data_logger1, data_logger2, data_logger3]

    assert len(data_logger1.euclidean_distance) == \
        len(data_logger2.euclidean_distance) == \
        len(data_logger3.euclidean_distance)

    # Below code is for
    # 1) A graph with SU's euclidean distance between real and predicted points.
    #     One legend will be Mittendorfer's method and another will be ours
    plt.plot(data_logger1.euclidean_distance, "-b", label=method_names[1])
    plt.plot(data_logger2.euclidean_distance, "-r", label=method_names[2])
    plt.plot(data_logger3.euclidean_distance, "-g", label=method_names[3])
    plt.title("L2 norms of each SU from original SU location")
    plt.xlabel("IMU number starting from 1")
    plt.ylabel("L2 Norm")
    plt.legend(loc="upper left")
    plt.xticks(np.arange(len(data_logger1.euclidean_distance)),
               np.arange(1, len(data_logger1.euclidean_distance) + 1))

    save_plt(l2_norm_plot_file_name)
    plt.show()
    add_image_to_md(filename=stats_file_name, img_filename=l2_norm_plot_file_name)

    column_names = [f"SU{i_su}" for i_su in range(1, n_su+1)]

    target_measure = "euclidean_distance"
    table = create_cascaded_table(
        method_names=method_names,
        data_loggers=data_loggers,
        target_measure=target_measure,
        column_names=column_names,
        row_names=["L2 Norm"])
    add_table_to_md(filename=stats_file_name, table=table,
                    headers=[target_measure]+column_names)

    # Print all average euclidean distances
    key = "average_euclidean_distance"
    with open(stats_file_name, "a") as f:
        for method_name, data_logger in zip(method_names, data_loggers):
            f.write(f"Method={method_name}, {key}={getattr(data_logger, key)}")
            f.write("\n")
            f.write("\n")

    # Below code is for
    # 1) A table comparing the dh params individually of our method and the others
    target_measure = "params"
    table = create_cascaded_table(
        method_names=method_names,
        data_loggers=data_loggers,
        target_measure=target_measure,
        column_names=column_names,
        row_names=["Θ<sub>0</sub>", "d<sub>0</sub>", "Θ<sub>acc</sub>", "d<sub>acc</sub>", "a<sub>acc</sub>", "α<sub>acc</sub>"])
    add_table_to_md(filename=stats_file_name, table=table,
                    headers=[target_measure]+column_names)

    # Below code is for
    # 1) A table comparing the orientations individually of our method and the others
    target_measure = "orientation"
    table = create_cascaded_table(
        method_names=method_names,
        data_loggers=data_loggers,
        target_measure=target_measure,
        column_names=column_names,
        row_names=["w", "x", "y", "z"])
    add_table_to_md(filename=stats_file_name, table=table,
                    headers=[target_measure]+column_names)

    plt.close()
    print("Generating statistics done!")
