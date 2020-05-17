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
su_joint_dict = {i: i for i in range(n_joint)}

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
        KinematicChain(
            n_joint=n_joint,
            su_joint_dict=su_joint_dict,
            bound_dict=bound_dict,
            linkdh_dict=None,
            sudh_dict=None)

        KinematicChain(
            n_joint=n_joint,
            su_joint_dict=su_joint_dict,
            bound_dict=bound_dict,
            linkdh_dict=linkdh_dict,
            sudh_dict=sudh_dict)

    def test_eval_poses(self):
        """
        You can set an origin poses by either by,
        1. Pass an argument eval_poses
        2. Call kinematic_chain.set_n_poses(poses)

        You can add a pose to just 1 joint by calling
        3. kinematic_chain.add_a_pose(i_joint, pose)

        This function is slightly faster than add_n_poses

        If you pass an argument, it will keep the poses as it is
        even if reset_poses() are called.
        On the other hand, set_n_poses() and add_a_pose() are only
        effective until reset_pose() is called.
        """
        eval_poses = np.array([0, 0, 0, -0.0698, 0, 0, 0])

        kinematic_chain = KinematicChain(
            n_joint=n_joint,
            su_joint_dict=su_joint_dict,
            bound_dict=bound_dict,
            linkdh_dict=linkdh_dict,
            sudh_dict=sudh_dict,
            eval_poses=eval_poses)

        np.testing.assert_array_equal(
            x=kinematic_chain.current_poses,
            y=np.zeros(7))
        np.testing.assert_array_equal(
            x=kinematic_chain.eval_poses,
            y=eval_poses)

        kinematic_chain.reset_poses()

        np.testing.assert_array_equal(
            x=kinematic_chain.current_poses,
            y=np.zeros(7))
        np.testing.assert_array_equal(
            x=kinematic_chain.eval_poses,
            y=eval_poses)

    def test_add_poses(self):
        poses = np.random.rand(7)

        kinematic_chain = KinematicChain(
            n_joint=n_joint,
            su_joint_dict=su_joint_dict,
            bound_dict=bound_dict,
            linkdh_dict=linkdh_dict,
            sudh_dict=sudh_dict)

        kinematic_chain.set_poses(poses)

        np.testing.assert_array_equal(
            x=kinematic_chain.current_poses,
            y=poses)

        kinematic_chain.reset_poses()
        # Since eval_poses were not passed in the constructor
        # The reset poses would be all 0s.
        np.testing.assert_array_equal(
            x=kinematic_chain.current_poses,
            y=np.zeros(7))

    def test_compute_eval_joint_TM(self):
        eval_poses = np.array([0, 0, 0, -0.0698, 0, 0, 0])

        kinematic_chain = KinematicChain(
            eval_poses=eval_poses,
            n_joint=n_joint,
            su_joint_dict=su_joint_dict,
            bound_dict=bound_dict,
            linkdh_dict=linkdh_dict,
            sudh_dict=sudh_dict)

        expected_positions = np.array([
            [0.000, 0.000, 0.333],
            [0.000, 0.000, 0.333],
            [0.000, -0.000, 0.649],
            [0.083, -0.000, 0.649],
            [0.027, -0.000, 1.038],
            [0.027, -0.000, 1.038],
            [0.115, -0.000, 1.032]
        ])

        expected_orientations = np.array([
            [0.000, 0.000, -0.000, 1.000],
            [-0.707, 0.000, 0.000, 0.707],
            [0.000, 0.000, -0.000, 1.000],
            [0.707, 0.025, -0.025, 0.707],
            [-0.000, 0.035, -0.000, 0.999],
            [0.707, 0.025, -0.025, 0.707],
            [0.999, -0.000, -0.035, 0.000]
        ])

        for i in range(n_joint):
            T = kinematic_chain.compute_joint_TM(i, pose_type='eval')
            np.testing.assert_array_almost_equal(
                x=T.position,
                y=expected_positions[i],
                decimal=2,
                err_msg=f"Joint {i+1}: \
                    \n expected: {expected_positions[i]} \
                    \n but got:  {T.position}"
            )

            q = expected_orientations[i]
            q = pyqt.Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            d = pyqt.Quaternion.absolute_distance(T.q, q)
            self.assertTrue(d < 0.01,
                            msg=f"Joint {i+1}: \
                                \n expected: {expected_orientations[i]} \
                                \n but got:  {T.quaternion}")

    def test_compute_eval_su_TM(self):
        eval_poses = np.array([0, 0, 0, -0.0698, 0, 0, 0])

        kinematic_chain = KinematicChain(
            eval_poses=eval_poses,
            n_joint=n_joint,
            su_joint_dict=su_joint_dict,
            bound_dict=bound_dict,
            linkdh_dict=linkdh_dict,
            sudh_dict=sudh_dict)

        for i in range(n_joint):
            T = kinematic_chain.compute_su_TM(i, pose_type='eval')
            expected_position = su_pose[f'su{i+1}']['position']  # noqa: E999
            q = su_pose[f'su{i+1}']['rotation']  # noqa: E999
            expected_orientation = pyqt.Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

            np.testing.assert_array_almost_equal(
                x=T.position,
                y=expected_position,
                decimal=2,
                err_msg=f'{i+1}th SU Position supposed to be {expected_position}')

            d = pyqt.Quaternion.absolute_distance(T.q, expected_orientation)
            self.assertTrue(
                expr=d < 0.01,
                msg=f'{i+1}th SU Orientation supposed to be {expected_orientation} \
                    but got {T.q}')

    def test_compute_current_joint_TM(self):
        current_poses = np.array([1.5708, 0.0, 1.5708, -1.5708, 1.5689, 1.5707, 1.5708])

        kinematic_chain = KinematicChain(
            n_joint=n_joint,
            su_joint_dict=su_joint_dict,
            bound_dict=bound_dict,
            linkdh_dict=linkdh_dict,
            sudh_dict=sudh_dict)

        # Apply joint poses
        kinematic_chain.set_poses(current_poses)

        # Curren Poses should be set correctly
        np.testing.assert_array_equal(
            x=kinematic_chain.current_poses,
            y=current_poses)

        expected_positions = np.array([
            [0.000, 0.000, 0.333],
            [0.000, 0.000, 0.333],
            [0.000, -0.000, 0.649],
            [-0.082, -0.000, 0.649],
            [-0.467, -0.000, 0.731],
            [-0.467, -0.000, 0.731],
            [-0.555, -0.000, 0.731]
        ])

        expected_orientations = np.array([
            [0.000, 0.000, 0.707, 0.707],
            [-0.500, -0.500, 0.500, 0.500],
            [0.000, -0.000, 1.000, -0.000],
            [-0.500, 0.500, 0.500, 0.500],
            [0.500, -0.500, -0.500, 0.500],
            [-0.000, 1.000, 0.001, 0.000],
            [0.500, 0.500, -0.500, 0.500],
        ])

        # set_poses should affect the current
        # Joint TransformationMatrix
        for i in range(n_joint):
            T = kinematic_chain.compute_joint_TM(i, pose_type='current')
            # Test Joint Positions
            np.testing.assert_array_almost_equal(
                x=T.position,
                y=expected_positions[i],
                decimal=2,
                err_msg=f"Joint {i+1}: \
                    \n expected: {expected_positions[i]} \
                    \n but got:  {T.position}"
            )
            # Test Joint Orientations
            q = expected_orientations[i]
            q = pyqt.Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            d = pyqt.Quaternion.absolute_distance(T.q, q)
            self.assertTrue(d < 0.01,
                            msg=f"Joint {i+1}: \
                                \n expected: {expected_orientations[i]} \
                                \n but got:  {T.quaternion}")

        # Reset current poses
        kinematic_chain.reset_poses()
        # Current poses should be reset to origin poses
        np.testing.assert_array_equal(
            x=kinematic_chain.current_poses,
            y=np.zeros(7))

    def test_compute_current_su_TM(self):
        current_poses = np.array([1.5708, 0.0, 1.5708, -1.5708, 1.5689, 1.5707, 1.5708])

        kinematic_chain = KinematicChain(
            n_joint=n_joint,
            su_joint_dict=su_joint_dict,
            bound_dict=bound_dict,
            linkdh_dict=linkdh_dict,
            sudh_dict=sudh_dict)

        # Apply joint poses
        kinematic_chain.set_poses(current_poses)

        # Curren Poses should be set correctly
        np.testing.assert_array_equal(
            x=kinematic_chain.current_poses,
            y=current_poses)

        expected_positions = np.array([
            [-0.000, 0.050, 0.183],
            [-0.060, 0.060, 0.333],
            [-0.000, 0.050, 0.569],
            [-0.083, 0.080, 0.709],
            [-0.367, -0.000, 0.831],
            [-0.417, 0.000, 0.701],
            [-0.554, 0.000, 0.681],
        ])

        expected_orientations = np.array([
            [-0.500, 0.500, 0.500, 0.500],
            [-0.707, -0.000, 0.000, 0.707],
            [-0.000, 0.707, 0.707, -0.000],
            [-0.000, 0.000, 0.707, 0.707],
            [0.001, 0.000, -0.707, 0.707],
            [0.001, 0.707, 0.001, 0.707],
            [0.707, 0.707, 0.000, -0.000],
        ])

        # set_poses should affect the current
        # Joint TransformationMatrix
        for i in range(n_joint):
            T = kinematic_chain.compute_su_TM(i, pose_type='current')
            # Test SU Positions
            np.testing.assert_array_almost_equal(
                x=T.position,
                y=expected_positions[i],
                decimal=2,
                err_msg=f"Joint {i+1}: \
                    \n expected: {expected_positions[i]} \
                    \n but got:  {T.position}"
            )
            # Test SU Orientations
            q = expected_orientations[i]
            q = pyqt.Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            d = pyqt.Quaternion.absolute_distance(T.q, q)
            self.assertTrue(d < 0.01,
                            msg=f"Joint {i+1}: \
                                \n expected: {expected_orientations[i]} \
                                \n but got:  {T.quaternion}")

        # Reset current poses
        kinematic_chain.reset_poses()
        # Current poses should be reset to origin poses
        np.testing.assert_array_equal(
            x=kinematic_chain.current_poses,
            y=np.zeros(7))

    def test_set_linkdh(self):
        eval_poses = np.array([0, 0, 0, -0.0698, 0, 0, 0])

        # Removed linkdh_dict
        kinematic_chain = KinematicChain(
            eval_poses=eval_poses,
            n_joint=n_joint,
            su_joint_dict=su_joint_dict,
            bound_dict=bound_dict,
            sudh_dict=sudh_dict)

        expected_positions = np.array([
            [0.000, 0.000, 0.333],
            [0.000, 0.000, 0.333],
            [0.000, -0.000, 0.649],
            [0.083, -0.000, 0.649],
            [0.027, -0.000, 1.038],
            [0.027, -0.000, 1.038],
            [0.115, -0.000, 1.032]
        ])

        expected_orientations = np.array([
            [0.000, 0.000, -0.000, 1.000],
            [-0.707, 0.000, 0.000, 0.707],
            [0.000, 0.000, -0.000, 1.000],
            [0.707, 0.025, -0.025, 0.707],
            [-0.000, 0.035, -0.000, 0.999],
            [0.707, 0.025, -0.025, 0.707],
            [0.999, -0.000, -0.035, 0.000]
        ])

        for i in range(n_joint):
            # First check if the DH Parameter Estimation is not correct
            T = kinematic_chain.compute_joint_TM(i, pose_type='eval')

            # Test Joint Positions
            np.testing.assert_raises(
                AssertionError,
                np.testing.assert_array_almost_equal,
                T.position,
                expected_positions[i])
            # Test Joint Orientations
            q = expected_orientations[i]
            q = pyqt.Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            d = pyqt.Quaternion.absolute_distance(T.q, q)
            self.assertFalse(d < 0.01)

            # Set Link DH Parameters
            kinematic_chain.set_linkdh(i, np.array(linkdh_dict[f'joint{i+1}']))

            # Now DH Parameter Estimation should be correct
            T = kinematic_chain.compute_joint_TM(i, pose_type='eval')
            # Test Joint Positions
            np.testing.assert_array_almost_equal(
                x=T.position,
                y=expected_positions[i],
                decimal=2,
                err_msg=f"Joint {i+1}: \
                    \n expected: {expected_positions[i]} \
                    \n but got:  {T.position}"
            )
            # Test Joint Orientations
            q = expected_orientations[i]
            q = pyqt.Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            d = pyqt.Quaternion.absolute_distance(T.q, q)
            self.assertTrue(d < 0.01,
                            msg=f"Joint {i+1}: \
                                \n expected: {expected_orientations[i]} \
                                \n but got:  {T.quaternion}")

    def test_set_sudh(self):
        eval_poses = np.array([0, 0, 0, -0.0698, 0, 0, 0])

        # Removed linkdh_dict
        kinematic_chain = KinematicChain(
            eval_poses=eval_poses,
            n_joint=n_joint,
            su_joint_dict=su_joint_dict,
            bound_dict=bound_dict,
            linkdh_dict=linkdh_dict)

        for i in range(n_joint):
            # First check if the DH Parameter Estimation is not correct
            T = kinematic_chain.compute_su_TM(i, pose_type='eval')
            expected_position = su_pose[f'su{i+1}']['position']  # noqa: E999
            expected_orientation = su_pose[f'su{i+1}']['rotation']  # noqa: E999

            # Test SU Positions
            np.testing.assert_raises(
                AssertionError,
                np.testing.assert_array_almost_equal,
                T.position,
                expected_position)
            # Test SU Orientations
            q = expected_orientation
            q = pyqt.Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            d = pyqt.Quaternion.absolute_distance(T.q, q)
            self.assertFalse(d < 0.01)

            # Set SU DH Parameters
            kinematic_chain.set_sudh(i, np.array(sudh_dict[f'su{i+1}']))

            # Now DH Parameter Estimation should be correct
            T = kinematic_chain.compute_su_TM(i, pose_type='eval')
            # Test SU Positions
            np.testing.assert_array_almost_equal(
                x=T.position,
                y=expected_position,
                decimal=2,
                err_msg=f"Joint {i+1}: \
                    \n expected: {expected_position} \
                    \n but got:  {T.position}"
            )
            # Test SU Orientations
            q = expected_orientation
            q = pyqt.Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            d = pyqt.Quaternion.absolute_distance(T.q, q)
            self.assertTrue(d < 0.01,
                            msg=f"Joint {i+1}: \
                                \n expected: {expected_orientation} \
                                \n but got:  {T.quaternion}")

    def test_add_a_pose_with_SU_pose(self):
        current_poses = np.array([1.5708, 0.0, 1.5708, -1.5708, 1.5689, 1.5707, 1.5708])

        kinematic_chain = KinematicChain(
            n_joint=n_joint,
            su_joint_dict=su_joint_dict,
            bound_dict=bound_dict,
            linkdh_dict=linkdh_dict,
            sudh_dict=sudh_dict)

        expected_positions = np.array([
            [-0.000, 0.050, 0.183],
            [-0.060, 0.060, 0.333],
            [-0.000, 0.050, 0.569],
            [-0.083, 0.080, 0.709],
            [-0.367, -0.000, 0.831],
            [-0.417, 0.000, 0.701],
            [-0.554, 0.000, 0.681],
        ])

        expected_orientations = np.array([
            [-0.500, 0.500, 0.500, 0.500],
            [-0.707, -0.000, 0.000, 0.707],
            [-0.000, 0.707, 0.707, -0.000],
            [-0.000, 0.000, 0.707, 0.707],
            [0.001, 0.000, -0.707, 0.707],
            [0.001, 0.707, 0.001, 0.707],
            [0.707, 0.707, 0.000, -0.000],
        ])

        # set_poses should affect the current
        # Joint TransformationMatrix
        dof_T_dof, rs_T_dof = kinematic_chain.get_current_TMs()
        for i in range(n_joint):
            # Add pose to each joint
            kinematic_chain.add_a_pose(
                i_joint=i,
                pose=current_poses[i],
                dof_T_dof=dof_T_dof,
                rs_T_dof=rs_T_dof)

            # Get TEMP SU Position
            T = kinematic_chain._compute_su_TM(i, dof_T_dof, rs_T_dof)
            # Test SU Positions
            np.testing.assert_array_almost_equal(
                x=T.position,
                y=expected_positions[i],
                decimal=2,
                err_msg=f"Joint {i+1}: \
                    \n expected: {expected_positions[i]} \
                    \n but got:  {T.position}"
            )
            # Test SU Orientations
            q = expected_orientations[i]
            q = pyqt.Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            d = pyqt.Quaternion.absolute_distance(T.q, q)
            self.assertTrue(d < 0.01,
                            msg=f"Joint {i+1}: \
                                \n expected: {expected_orientations[i]} \
                                \n but got:  {T.quaternion}")

        # Reset current poses
        kinematic_chain.reset_poses()
        # Current poses should be reset to origin poses
        np.testing.assert_array_equal(
            x=kinematic_chain.current_poses,
            y=np.zeros(7))

    def test_compute_params_at(self):
        eval_poses = np.array([0, 0, 0, -0.0698, 0, 0, 0])

        # Removed linkdh_dict
        kinematic_chain = KinematicChain(
            n_joint=n_joint,
            su_joint_dict=su_joint_dict,
            bound_dict=bound_dict,
            eval_poses=eval_poses)

        for i in range(n_joint):
            # Set the DH parameters
            params, bounds = kinematic_chain.get_params_at(i)
            for parameter, bound in zip(params, bounds):
                self.assertTrue(bound[0] <= parameter <= bound[1])

    def test_set_params_at(self):
        eval_poses = np.array([0, 0, 0, -0.0698, 0, 0, 0])

        # Removed linkdh_dict
        kinematic_chain = KinematicChain(
            n_joint=n_joint,
            su_joint_dict=su_joint_dict,
            bound_dict=bound_dict,
            eval_poses=eval_poses)

        for i in range(n_joint):
            # Get the correct DH parameters
            params = linkdh_dict[f'joint{i+1}'] + sudh_dict[f'su{i+1}']
            # Set the DH parameters
            kinematic_chain.set_params_at(i, np.array(params))

            expected_position = su_pose[f'su{i+1}']['position']  # noqa: E999
            expected_orientation = su_pose[f'su{i+1}']['rotation']  # noqa: E999

            # Now DH Parameter Estimation should be correct
            T = kinematic_chain.compute_su_TM(i, pose_type='eval')
            # Test SU Positions
            np.testing.assert_array_almost_equal(
                x=T.position,
                y=expected_position,
                decimal=2,
                err_msg=f"Joint {i+1}: \
                    \n expected: {expected_position} \
                    \n but got:  {T.position}"
            )
            # Test SU Orientations
            q = expected_orientation
            q = pyqt.Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            d = pyqt.Quaternion.absolute_distance(T.q, q)
            self.assertTrue(d < 0.01,
                            msg=f"Joint {i+1}: \
                                \n expected: {expected_orientation} \
                                \n but got:  {T.quaternion}")


if __name__ == '__main__':
    unittest.main()
