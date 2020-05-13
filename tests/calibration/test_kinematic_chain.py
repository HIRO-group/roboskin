"""
Testing utils module
"""
import os
import unittest
import numpy as np
import pyquaternion as pyqt
from robotic_skin.calibration.utils.io import load_robot_configs
from robotic_skin.calibration.kinematic_chain import KinematicChain

repodir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
robot_config = load_robot_configs(os.path.join(repodir, 'config'), 'panda')

linkdh_dict = robot_config['dh_parameter']
sudh_dict = robot_config['su_dh_parameter']
su_pose = robot_config['su_pose']

n_joint = len(linkdh_dict)
su_joint_dict = {i+1: i+1 for i in range(n_joint)}

bounds = np.array([
    [-np.pi, np.pi],    # th
    [0.0, 1.0],         # d
    [0.0, 1.0],         # a
    [-np.pi, np.pi]])   # alpha
bounds_su = np.array([
    [-np.pi, np.pi],    # th
    [-1.0, 1.0],        # d
    [-np.pi, np.pi],    # th
    [0.0, 0.2],         # d
    [0.0, 0.0001],      # a     # 0 gives error
    [0, np.pi]])        # alpha
bound_dict = {'link': bounds, 'su': bounds_su}


class KinematicChainTest(unittest.TestCase):
    """
    KinematicChain Matrix Test Class
    """
    def test_constructor(self):
        """
        """
        kinematic_chain = KinematicChain(
            n_joint=n_joint,
            su_joint_dict=su_joint_dict,
            bound_dict=bound_dict,
            linkdh_dict=None,
            sudh_dict=None)

        kinematic_chain = KinematicChain(
            n_joint=n_joint,
            su_joint_dict=su_joint_dict,
            bound_dict=bound_dict,
            linkdh_dict=linkdh_dict,
            sudh_dict=sudh_dict)

    def test_get_origin_joint_TM(self):
        kinematic_chain = KinematicChain(
            n_joint=n_joint,
            su_joint_dict=su_joint_dict,
            bound_dict=bound_dict,
            linkdh_dict=linkdh_dict,
            sudh_dict=sudh_dict)

        Ts = kinematic_chain.rs_T_dof

        for i in range(n_joint):
            T = kinematic_chain.get_origin_joint_TM(i+1)

    def test_get_origin_su_TM(self):
        init_poses = [0, 0, 0, -0.0698, 0, 0, 0]

        kinematic_chain = KinematicChain(
            init_poses=init_poses,
            n_joint=n_joint,
            su_joint_dict=su_joint_dict,
            bound_dict=bound_dict,
            linkdh_dict=linkdh_dict,
            sudh_dict=sudh_dict)

        # kinematic_chain.add_a_pose(4, -0.0698)

        for i in range(n_joint):
            T = kinematic_chain.get_current_su_TM(i+1)
            expected_position = su_pose[f'su{i+1}']['position']  # noqa: E999
            q = su_pose[f'su{i+1}']['rotation']  # noqa: E999
            expected_rotation = pyqt.Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

            np.testing.assert_array_almost_equal(
                x=T.position,
                y=expected_position,
                decimal=2,
                err_msg=f'{i+1}th SU Position supposed to be {expected_position}')

            d = pyqt.Quaternion.absolute_distance(T.q, expected_rotation)
            self.assertTrue(
                expr=d < 0.01,
                msg=f'{i+1}th SU Rotation supposed to be {expected_rotation} \
                    but got {T.q}')


if __name__ == '__main__':
    unittest.main()
