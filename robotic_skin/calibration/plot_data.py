import os
import rospkg
import pickle
import numpy as np
import matplotlib.pyplot as plt

from robotic_skin.calibration.utils import TransMat, ParameterManager

PACKAGE_HOME_DIR = os.path.abspath(__file__ + "/../../../")


def load_data(robot='panda'):
    directory = os.path.join(rospkg.RosPack().get_path('ros_robotic_skin'), 'data')

    filename = '_'.join(['constant_data', robot])
    filepath = os.path.join(directory, filename + '.pickle')
    with open(filepath, 'rb') as f:
        constant = pickle.load(f, encoding='latin1')

    return constant


def load_dhparams(robot='panda'):
    # th, d, a, al
    if robot == 'sawyer':
        dhparams = np.array([
            [0,         0.317,      0,      0],
            [np.pi/2,   0.1925,     0.081,  -np.pi/2],
            [0,         0.4,        0,      np.pi/2],
            [0,         -0.1685,    0,      -np.pi/2],
            [0,         0.4,        0,      np.pi/2],
            [0,         0.1363,     0,      -np.pi/2],
            [np.pi,     0.13375,    0,      np.pi/2]
        ])
    else:
        dhparams = np.array([
            [0, 0.333,  0,          0],
            [0, 0,      0,          -np.pi/2],
            [0, 0.316,  0,          np.pi/2],
            [0, 0,      0.0825,     np.pi/2],
            [0, 0.384,  -0.0825,    -np.pi/2],
            [0, 0,      0,          np.pi/2],
            [0, 0,      0.088,      np.pi/2]
        ])

    return dhparams


def estimate_acceleration_analytically(Tdofs, Tjoints, Tdofi2su, d, i, curr_w):
    # Transformation Matrix from su to rs in rs frame
    rs_T_su = TransMat(np.zeros(4))
    # Transformation Matrix from the last DoFi to the excited DoFd
    dofd_T_dofi = TransMat(np.zeros(4))

    for j in range(d+1):
        # print(j)
        rs_T_su = rs_T_su.dot(Tdofs[j]).dot(Tjoints[j])

    for j in range(d+1, i+1):
        # print(j, d, i)
        rs_T_su = rs_T_su.dot(Tdofs[j]).dot(Tjoints[j])
        dofd_T_dofi = dofd_T_dofi.dot(Tdofs[j]).dot(Tjoints[j])

    rs_T_su = rs_T_su.dot(Tdofi2su)
    dof_T_su = dofd_T_dofi.dot(Tdofi2su)

    dofd_r_su = dof_T_su.position
    # Every joint rotates along its own z axis
    w_dofd = np.array([0, 0, curr_w])
    a_dofd = np.cross(w_dofd, np.cross(w_dofd, dofd_r_su))
    # print('[From Joint{} to IMU{}, w={}] r = {}    a = {}'.format(d, i, n2s(w_dofd), n2s(dofd_r_su,3), n2s(a_dofd,3)))

    g_rs = np.array([0, 0, 9.8])

    a_su = np.dot(dof_T_su.R.T, a_dofd) + np.dot(rs_T_su.R.T, g_rs)
    # a_su = np.dot(rs_T_su.R.T, g_rs)

    return a_su


def get_su_transmat(i, robot='panda'):
    if robot == 'saywer':
        params = np.array([
            [1.57,  -0.157,  -1.57, 0.07, 0,  1.57],
            [-1.57, -0.0925, 1.57,  0.07, 0,  1.57],
            [-1.57, -0.16,   1.57,  0.05, 0,  1.57],
            [-1.57, 0.0165,  1.57,  0.05, 0,  1.57],
            [-1.57, -0.17,   1.57,  0.05, 0,  1.57],
            [-1.57, 0.0053,  1.57,  0.04, 0,  1.57],
            [0.0,   0.12375, 0.0,   0.03, 0, -1.57]
        ])
    elif robot == 'panda':
        params = np.array([
            [1.57, -0.15, -1.57, 0.05, 0, 1.57],
            [1.57, 0.06, -1.57, 0.06, 0, 1.57],
            [0, -0.08, 0, 0.05, 0, 1.57],
            [-1.57, 0.08, 1.57, 0.06, 0, 1.57],
            [3.14, -0.1, 3.14, 0.1, 0, 1.57],
            [-1.57, 0.03, 1.57, 0.05, 0, 1.57],
            [1.57, 0, -1.57, 0.05, 0, 1.57]
        ])
    else:
        raise NotImplementedError("Define a robot's DH Parameters")

    Tdof2vdof = TransMat(params[i, :2])
    Tvdof2su = TransMat(params[i, 2:])

    return Tdof2vdof.dot(Tvdof2su)


if __name__ == '__main__':
    Data = load_data()

    pose_names = list(Data.keys())
    joint_names = list(Data[pose_names[0]].keys())
    imu_names = list(Data[pose_names[0]][joint_names[0]].keys())
    n_pose = len(pose_names)
    n_joint = len(joint_names)

    bounds = np.array([
        [0.0, 0.00001],     # th
        [-1.0, 1.0],        # d
        [-0.2, 0.2],        # a     (radius)
        [-np.pi, np.pi]])   # alpha
    bounds_su = np.array([
        [-np.pi, np.pi],    # th
        [-1.0, 1.0],        # d
        [-np.pi, np.pi],    # th
        [0.0, 0.2],         # d
        [0.0, 0.0001],      # a     # 0 gives error
        [0, np.pi]])        # alpha
    dhparams = load_dhparams()

    param_manager = ParameterManager(n_joint, bounds, bounds_su, dhparams)

    images_dir = os.path.join(PACKAGE_HOME_DIR, 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    for i, imu_name in enumerate(imu_names):
        Tdofs = param_manager.get_tmat_until(i)
        Tdof2su = get_su_transmat(i)

        for p in range(n_pose):
            # for d in range(max(0, i-2), i+1):
            for d in range(i+1):
                data = Data[pose_names[p]][joint_names[d]][imu_names[i]][0]
                meas_accels = data[:, 4:7]
                joints = data[:, 7:14]
                angular_velocities = data[:, 14]

                model_accels = []
                for meas_accel, joint, curr_w in zip(meas_accels, joints, angular_velocities):
                    Tjoints = [TransMat(joint) for joint in joint[:i+1]]
                    model_accel = estimate_acceleration_analytically(Tdofs, Tjoints, Tdof2su, d, i, curr_w)
                    model_accels.append(model_accel)
                model_accels = np.array(model_accels)

                if len(model_accels) == 0:
                    print('Position%i' % (p), 'Joint%i' % (d), 'IMU%i' % (i), model_accels)
                    print(data)
                else:
                    print('Position%i' % (p), 'Joint%i' % (d), 'IMU%i' % (i))

                    A_measured = np.linalg.norm(meas_accels, axis=1)
                    A_model = np.linalg.norm(model_accels, axis=1)

                    fig = plt.figure(figsize=(16, 12))
                    gs = fig.add_gridspec(3, 2)

                    ax = fig.add_subplot(gs[0, 0])
                    ax.plot(np.arange(meas_accels.shape[0]), A_measured, label='Measured')
                    ax.plot(np.arange(model_accels.shape[0]), A_model, label='Estimated')
                    ax.hlines(y=9.81, xmin=0, xmax=meas_accels.shape[0], color='r', label='G=9.81')
                    ax.set_title('Acceleration Norm')
                    ax.set_ylim([9, 11])
                    ax.legend()

                    ax = fig.add_subplot(gs[0, 1])
                    ax.plot(np.arange(meas_accels.shape[0]), meas_accels[:, 0], label='Measured')
                    ax.plot(np.arange(model_accels.shape[0]), model_accels[:, 0], label='Estimated')
                    ax.set_title('Acceleration for x axis')
                    # ax.set_ylim([-11, 11])
                    ax.legend()

                    ax = fig.add_subplot(gs[1, 0])
                    ax.plot(np.arange(meas_accels.shape[0]), meas_accels[:, 1], label='Measured')
                    ax.plot(np.arange(model_accels.shape[0]), model_accels[:, 1], label='Estimated')
                    ax.set_title('Acceleration for y axis')
                    # ax.set_ylim([-11, 11])
                    ax.legend()

                    ax = fig.add_subplot(gs[1, 1])
                    ax.plot(np.arange(meas_accels.shape[0]), meas_accels[:, 2], label='Measured')
                    ax.plot(np.arange(model_accels.shape[0]), model_accels[:, 2], label='Estimated')
                    ax.set_title('Acceleration for z axis')
                    # ax.set_ylim([-11, 11])
                    ax.legend()

                    ax = fig.add_subplot(gs[2, :])
                    ax.plot(np.arange(angular_velocities.size), angular_velocities)
                    ax.set_title('Joint Angular Velocity [rad/s]')
                    if np.sum(angular_velocities) >= 0:
                        ax.set_ylim([0, 1.1])
                    else:
                        ax.set_ylim([-1.1, 0])

                    savepath = os.path.join(images_dir, 'max_accel_Pose%i_Joint%i_IMU%i.png' % (p, d, i))
                    print(savepath)
                    plt.savefig(savepath)
                    plt.close()
