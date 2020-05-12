"""
Testing utils module
"""
import unittest
import numpy as np
import pyquaternion as pyqt
from robotic_skin.calibration.utils.quaternion import pyquat_to_numpy
from robotic_skin.calibration.transformation_matrix import TransformationMatrix as TM

N_JOINT = 7
INIT_POSE = np.zeros(N_JOINT)
SECOND_POSE = (np.pi/2)*np.ones(N_JOINT)
BOUNDS = np.array([
    [-np.pi, np.pi],    # th
    [0.0, 1.0],         # d
    [0.0, 1.0],         # a
    [-np.pi, np.pi]])   # alpha
BOUNDS_SU = np.array([
    [-np.pi, np.pi],    # th
    [-1.0, 1.0],        # d
    [-np.pi, np.pi],    # th
    [0.0, 0.2],         # d
    [0.0, 0.0001],      # a     # 0 gives error
    [0, np.pi]])        # alpha

PANDA_DHPARAMS = {'joint1': [0, 0.333, 0, 0],
                  'joint2': [0, 0, 0, -1.57079633],
                  'joint3': [0, 0.316, 0, 1.57079633],
                  'joint4': [0, 0, 0.0825, 1.57079633],
                  'joint5': [0, 0.384, -0.0825, -1.57079633],
                  'joint6': [0, 0, 0, 1.57079633],
                  'joint7': [0, 0, 0.088, 1.57079633]}

SAWYER_DHPARAMS = {'joint1': [0, 0.317, 0, 0],
                   'joint2': [1.57079633, 0.1925, 0.081, -1.57079633],
                   'joint3': [0, 0.4, 0, 1.57079633],
                   'joint4': [0, -0.1685, 0, -1.57079633],
                   'joint5': [0, 0.4, 0, 1.57079633],
                   'joint6': [0, 0.1363, 0, -1.57079633],
                   'joint7': [3.14159265, 0.13375, 0, 1.57079633]}


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
            T = TM.from_bounds(bounds=BOUNDS)
            for parameter, bound in zip(T.parameters, BOUNDS):
                self.assertTrue(bound[0] <= parameter <= bound[1])

    def test_list(self):
        """
        Test bounds of transformation matrix
        """
        l = [1, 2]
        T = TM.from_list(l, keys=['theta', 'd'])
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
        n_joint = 7
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
        np.testing.assert_array_almost_equal(T.q, q)

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
        np.testing.assert_array_almost_equal(T.q, q)

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
        np.testing.assert_array_almost_equal(T.q, q)

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
        np.testing.assert_array_almost_equal(T.q, q)

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
        np.testing.assert_array_almost_equal(T.q, q)

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
        np.testing.assert_array_almost_equal(T.q, q)

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
        T3 = T1*T2

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
        np.testing.assert_array_almost_equal(T3.q, q)

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
        joint_angles = [0, 0, 0, 0, 0, 0, 0]
        T_dofs = [TM.from_list(PANDA_DHPARAMS['joint%i' % (i+1)]) for i in range(7)]
        T_joints = [TM(theta=rad) for rad in joint_angles]
        dof_T_vdof = TM(theta=np.pi/2)
        vdof_T_su = TM.from_list([-np.pi/2, 0.05, 0, np.pi/2])

        # Tansform  from world to end-effector's IMU
        T = TM.from_numpy(np.zeros(4))
        for T_dof, T_joint in zip(T_dofs, T_joints):
            T = T * T_dof * T_joint

        print('=====')
        print(T.position, T.q)
        T = T * dof_T_vdof * vdof_T_su

        # Given by TF: Just run `rosrun tf tf_echo /world /imu_link6`
        # expected_position = [0.125, 0.020, 0.891]
        expected_position = [0.165, 0.000, 1.028]
        print(T.position, expected_position)
        np.testing.assert_array_almost_equal(T.position, expected_position, decimal=2)

        # DEG == 90
        joint_angles = [0, np.pi/2, 0, 0, 0, 0, 0]
        T_dofs = [TM.from_list(PANDA_DHPARAMS['joint%i' % (i+1)]) for i in range(7)]
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

    def test_sawyer_world_to_endeffector_su(self):
        """
        Uses Sawyer's DH Parameters to reach an IMU mounted on an end-effector
        Verifies if the transformation matches the expected IMU position
        """
        # DEG == 0
        joint_angles = [0, 0, 0, 0, 0, 0, 0]
        T_dofs = [TM.from_list(SAWYER_DHPARAMS['joint%i' % (i+1)]) for i in range(7)]
        T_joints = [TM(theta=rad) for rad in joint_angles]
        dof_T_vdof = TM(theta=0, d=0.1)
        vdof_T_su = TM.from_list([0, 0.03, 0, -np.pi/2])

        # Tansform  from world to end-effector's IMU
        T = TM.from_numpy(np.zeros(4))
        for T_dof, T_joint in zip(T_dofs, T_joints):
            T = T * T_dof * T_joint
        T = T * dof_T_vdof * vdof_T_su

        # Given by TF: Just run `rosrun tf tf_echo /world /imu_link6`
        expected_position = [1.084, 0.131, 0.195]
        np.testing.assert_array_almost_equal(T.position, expected_position, decimal=2)

        # DEG == 90
        joint_angles = [0, -np.pi/2, 0, 0, 0, 0, 0]
        T_dofs = [TM.from_list(SAWYER_DHPARAMS['joint%i' % (i+1)]) for i in range(7)]
        T_joints = [TM(theta=rad) for rad in joint_angles]
        dof_T_vdof = TM(theta=0, d=0.1)
        vdof_T_su = TM.from_list([0, 0.03, 0, -np.pi/2])

        # Tansform  from world to end-effector's IMU
        T = TM.from_numpy(np.zeros(4))
        for T_dof, T_joint in zip(T_dofs, T_joints):
            T = T * T_dof * T_joint
        T = T * dof_T_vdof * vdof_T_su

        # Given by TF: Just run `rosrun tf tf_echo /world /imu_link6`
        expected_position = [-0.043, 0.131, 1.319]
        np.testing.assert_array_almost_equal(T.position, expected_position, decimal=2)


if __name__ == '__main__':
    unittest.main()
