"""
Testing utils module
"""
import os
import unittest
import numpy as np
import pyquaternion as pyqt
from robotic_skin.calibration.utils.io import load_robot_configs
from robotic_skin.calibration.utils.quaternion import pyquat_to_numpy
from robotic_skin.calibration.transformation_matrix import TransformationMatrix as TM


repodir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
robot_config = load_robot_configs(os.path.join(repodir, 'config'), 'panda')

linkdh_dict = robot_config['dh_parameter']
sudh_dict = robot_config['su_dh_parameter']
su_pose = robot_config['su_pose']

n_joint = len(linkdh_dict)

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


class TransformationMatrixTest(unittest.TestCase):
    """
    Transformation Matrix Test Class
    """
    def test_matrix(self):
        """
        Test to create an identity matrix
        """
        T = TM.from_numpy(np.zeros(4))
        np.testing.assert_array_equal(T.matrix, np.eye(4))

    def test_n_params(self):
        """
        Test a constructor with different number of parameters
        """
        T = TM.from_numpy(np.random.rand(1), keys=['theta'])
        self.assertEqual(T.key_index, 0)

        T = TM.from_numpy(np.random.rand(2), keys=['theta', 'd'])
        np.testing.assert_array_equal(T.key_index, np.array([0, 1]))

        T = TM.from_numpy(np.random.rand(4))
        np.testing.assert_array_equal(T.key_index, np.array([0, 1, 2, 3]))

    def test_bound(self):
        """
        Test bounds of transformation matrix
        """
        iterations = 100

        for _ in range(iterations):
            T = TM.from_bounds(bounds=bound_dict['link'])
            for parameter, bound in zip(T.parameters, bound_dict['link']):
                self.assertTrue(bound[0] <= parameter <= bound[1])

    def test_list(self):
        """
        Test bounds of transformation matrix
        """
        T = TM.from_list([1, 2], keys=['theta', 'd'])
        expected = np.array([1., 2., 0., 0.])
        np.testing.assert_array_almost_equal(T.params, expected, decimal=1)

    def test_wrong_number_of_params(self):
        """
        Test if a constructor outputs error
        if other than 1, 2, 4 params are given
        """
        self.assertRaises(ValueError, TM.from_numpy, np.random.rand(3))

    def test_R_shape(self):
        """
        Test the shape of the resulting rotation matrix
        """
        T = TM.from_numpy(np.zeros(4))
        self.assertEqual(T.R.shape, np.zeros((3, 3)).shape)

    def test_position_shape(self):
        """
        Test the shape of the resulting positions
        """
        T = TM.from_numpy(np.zeros(4))
        self.assertEqual(T.position.shape, np.zeros(3).shape)

    def test_sub_position_into_ndarray(self):
        """
        Test to substitute an np.array to np.ndarray
        """
        positions = np.zeros((n_joint, 3))
        T = TM.from_numpy(np.zeros(4))

        raised = False
        try:
            positions[0, :] = T.position
        except Exception:
            raised = True

        self.assertFalse(raised, 'Exception raised')

    def test_rotation_around_z(self):
        """
        Test a tranformation matrix by rotating 90 degrees
        """
        # 90 Deg
        T = TM(theta=np.pi/2)
        expected_R = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        np.testing.assert_array_almost_equal(T.R, expected_R)

        q = pyqt.Quaternion(axis=[0, 0, 1], angle=np.pi/2)
        q = pyquat_to_numpy(q)
        np.testing.assert_array_almost_equal(T.quaternion, q)

        # -90 Deg
        T = TM(theta=-np.pi/2)
        expected_R = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1]
        ])
        np.testing.assert_array_almost_equal(T.R, expected_R)

        q = pyqt.Quaternion(axis=[0, 0, 1], angle=-np.pi/2)
        q = pyquat_to_numpy(q)
        np.testing.assert_array_almost_equal(T.quaternion, q)

        # 180 Deg
        T = TM(theta=np.pi)
        expected_R = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])
        np.testing.assert_array_almost_equal(T.R, expected_R)

        q = pyqt.Quaternion(axis=[0, 0, 1], angle=np.pi)
        q = pyquat_to_numpy(q)
        np.testing.assert_array_almost_equal(T.quaternion, q)

    def test_rotation_around_x(self):
        """
        Test a tranformation matrix by rotating 90 degrees
        """
        # 90 Deg
        T = TM(alpha=np.pi/2)
        expected_R = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        np.testing.assert_array_almost_equal(T.R, expected_R)

        q = pyqt.Quaternion(axis=[1, 0, 0], angle=np.pi/2)
        q = pyquat_to_numpy(q)
        np.testing.assert_array_almost_equal(T.quaternion, q)

        # -90 Deg
        T = TM(alpha=-np.pi/2)
        expected_R = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ])
        np.testing.assert_array_almost_equal(T.R, expected_R)

        q = pyqt.Quaternion(axis=[1, 0, 0], angle=-np.pi/2)
        q = pyquat_to_numpy(q)
        np.testing.assert_array_almost_equal(T.quaternion, q)

        # 180 Deg
        T = TM(alpha=np.pi)
        expected_R = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
        np.testing.assert_array_almost_equal(T.R, expected_R)

        q = pyqt.Quaternion(axis=[1, 0, 0], angle=np.pi)
        q = pyquat_to_numpy(q)
        np.testing.assert_array_almost_equal(T.quaternion, q)

    def test_dot_product(self):
        """
        Test the tranformation order.
        Checks whether it rotates T1 first and then T2.
        This also checks whether each TransformationMatrix rotates X first and then Z
        It should look like
            ^
            |    (2+sqrt(2), sqrt(2))
            |    /
            |   / 45 degrees (+ 2 )
            -------->
              2
        The resulting position should be [0, 2*sqrt(2)]
        """
        # 1. Translate x axis for 2
        # 2. Rotates 45 degrees + Translate z axis for 4
        T1 = TM.from_numpy(np.array([np.pi/4, 4, 2, 0]))
        # 1. Translate x axis for 2
        # 2. Rotates 90 degrees + Translate z axis for 4
        T2 = TM.from_numpy(np.array([np.pi/2, 4, 2, 0]))
        T3 = T1 * T2

        a = 1/np.sqrt(2)
        expected_R = np.array([
            [-a, -a, 0],
            [a, -a, 0],
            [0, 0, 1]
        ])
        np.testing.assert_array_almost_equal(T3.R, expected_R)

        expected_pos = np.array([2+np.sqrt(2), np.sqrt(2), 8])
        np.testing.assert_array_almost_equal(T3.position, expected_pos)

        q1 = pyqt.Quaternion(axis=[0, 0, 1], angle=np.pi/4)
        q2 = pyqt.Quaternion(axis=[0, 0, 1], angle=np.pi/2)
        q = pyquat_to_numpy(q1 * q2)
        np.testing.assert_array_almost_equal(T3.quaternion, q)

    def test_gravity_vector(self):
        """
        1. Rotate 90 deg around the same axis (for SU)
        2. Rotate 90 deg around X axis (for SU)
        """
        world_g = np.array([0, 0, 9.81])

        world_T_vdof = TM(theta=np.pi/2, d=0)
        vdof_T_su = TM.from_numpy(np.array([0, 0, 0, np.pi/2]))

        T = world_T_vdof * vdof_T_su
        su_R_world = T.R.T
        su_g = np.dot(su_R_world, world_g)

        expected_vec = np.array([0, 9.81, 0])

        np.testing.assert_array_almost_equal(su_g, expected_vec, decimal=2)

    def test_vector_rotation_with_joint_rotation(self):
        """
        1. Rotate 90 deg around the motor axis.
        2. Rotate 90 deg around the same axis (for SU)
        3. Rotate 90 deg around X axis (for SU)
        """
        world_vec = np.array([1, 2, 3])

        T_joint = TM(theta=np.pi/2)
        joint_T_vdof = TM(theta=np.pi/2, d=0)
        vdof_T_su = TM.from_numpy(np.array([0, 0, 0, np.pi/2]))

        T = T_joint * joint_T_vdof * vdof_T_su
        su_R_world = T.R.T
        su_g = np.dot(su_R_world, world_vec)

        expected_vec = np.array([-1, 3, 2])

        np.testing.assert_array_almost_equal(su_g, expected_vec, decimal=2)

    def test_panda_world_to_endeffector_su(self):
        """
        Uses Panda's DH Parameters to reach an IMU mounted on an end-effector
        Verifies if the transformation matches the expected IMU position
        """
        # DEG == 0
        joint_angles = [0, 0, 0, -0.0698, 0, 0, 0]

        # Tansform  from world to each SU coordinate
        # Verify if all the imus are in correct poses
        T = TM.from_numpy(np.zeros(4))
        for i, rad in enumerate(joint_angles):
            T_dof = TM.from_list(linkdh_dict['joint%i' % (i+1)])
            T_joint = TM(theta=rad)
            T = T * T_dof * T_joint

            su_str = f'su{i+1}'  # noqa: E999
            dof_T_vdof = TM.from_list(sudh_dict[su_str][:2], keys=['theta', 'd'])
            vdof_T_su = TM.from_list(sudh_dict[su_str][2:])
            rs_T_su = T * dof_T_vdof * vdof_T_su
            np.testing.assert_array_almost_equal(
                x=rs_T_su.position,
                y=su_pose[su_str]['position'],
                decimal=2)

            q = su_pose[su_str]['rotation']
            q = pyqt.Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            # We evaluate distance because quaternions cannot be directly compared,
            # since negation of the quaternion is equal to the original quaternion.
            # This is because negative rotation around the flipped axis is
            # basically equal to the original rotation.
            d = pyqt.Quaternion.absolute_distance(rs_T_su.q, q)
            self.assertTrue(d < 0.01)

        # DEG == 90
        joint_angles = [0, np.pi/2, 0, -0.0698, 0, 0, 0]
        T_dofs = [TM.from_list(linkdh_dict['joint%i' % (i+1)]) for i in range(7)]
        T_joints = [TM(theta=rad) for rad in joint_angles]
        dof_T_vdof = TM(theta=np.pi/4, d=0.14)
        vdof_T_su = TM.from_list([0, 0.03, 0, np.pi/2])

        # Tansform  from world to end-effector's IMU
        T = TM.from_numpy(np.zeros(4))
        for T_dof, T_joint in zip(T_dofs, T_joints):
            T = T * T_dof * T_joint
        T = T * dof_T_vdof * vdof_T_su

        # Given by TF: Just run `rosrun tf tf_echo /world /imu_link6`
        expected_position = [0.557, 0.020, 0.206]
        np.testing.assert_array_almost_equal(T.position, expected_position, decimal=2)


if __name__ == '__main__':
    unittest.main()
