"""
Testing utils module
"""
import unittest
import numpy as np
from robotic_skin.calibration.utils import TransMat, ParameterManager

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
    
PANDA_DHPARAMS = np.array([
    [0, 0.333,  0,          0],
    [0, 0,      0,          -np.pi/2],
    [0, 0.316,  0,          np.pi/2],
    [0, 0,      0.0825,     np.pi/2],
    [0, 0.384,  -0.0825,    -np.pi/2],
    [0, 0,      0,          np.pi/2],
    [0, 0,      0.088,      np.pi/2]
])

SAWYER_DHPARAMS = np.array([
    [0,         0.317,      0,      0],
    [np.pi/2,   0.1925,     0.081,  -np.pi/2],
    [0,         0.4,        0,      np.pi/2],
    [0,         -0.1685,    0,      -np.pi/2],
    [0,         0.4,        0,      np.pi/2],
    [0,         0.1363,     0,      -np.pi/2],
    [np.pi,     0.13375,    0,      np.pi/2]
])

class TransMatTest(unittest.TestCase):
    """
    Transformation Matrix Test Class
    """
    def test_matrix(self):
        """
        Test to create an identity matrix
        """
        T = TransMat(np.zeros(4))
        np.testing.assert_array_equal(T.mat, np.eye(4))

    def test_n_params(self):
        """
        Test a constructor with different number of parameters
        """
        T = TransMat(np.random.rand(1))
        self.assertEqual(T.n_params, 1)
        
        T = TransMat(np.random.rand(2))
        self.assertEqual(T.n_params, 2)

        T = TransMat(np.random.rand(4))
        self.assertEqual(T.n_params, 4)

    def test_bound(self):
        iterations = 100

        for i in range(iterations):
            T = TransMat(bounds=BOUNDS)
            for parameter, bound in zip(T.parameters, BOUNDS):
                self.assertTrue(bound[0] <= parameter <= bound[1])

    def test_wrong_number_of_params(self):
        """
        Test if a constructor outputs error 
        if other than 1, 2, 4 params are given
        """
        self.assertRaises(ValueError, TransMat, np.random.rand(3))

    def test_R_shape(self):
        """
        Test the shape of the resulting rotation matrix
        """
        T = TransMat(np.zeros(4))
        self.assertEqual(T.R.shape, np.zeros((3, 3)).shape)

    def test_position_shape(self):
        """
        Test the shape of the resulting positions 
        """
        T = TransMat(np.zeros(4))
        self.assertEqual(T.position.shape, np.zeros(3).shape)

    def test_sub_position_into_ndarray(self):
        """
        Test to substitute an np.array to np.ndarray
        """
        n_joint = 7
        positions = np.zeros((n_joint, 3))
        T = TransMat(np.zeros(4))

        raised = False
        try:
            positions[0, :] = T.position
        except:
            raised = True

        self.assertFalse(raised, 'Exception raised')

    def test_rotation_around_z(self):
        """
        Test a tranformation matrix by rotating 90 degrees
        """
        # 90 Deg
        T = TransMat(np.array([np.pi/2, 0, 0, 0]))
        expected_R = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        np.testing.assert_array_almost_equal(T.R, expected_R)
        
        # -90 Deg
        T = TransMat(np.array([-np.pi/2, 0, 0, 0]))
        expected_R = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1]
        ])
        np.testing.assert_array_almost_equal(T.R, expected_R)

        # 180 Deg
        T = TransMat(np.array([np.pi, 0, 0, 0]))
        expected_R = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])
        np.testing.assert_array_almost_equal(T.R, expected_R)

    def test_rotation_around_x(self):
        """
        Test a tranformation matrix by rotating 90 degrees
        """
        # 90 Deg
        T = TransMat(np.array([0, 0, 0, np.pi/2]))
        expected_R = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        np.testing.assert_array_almost_equal(T.R, expected_R)
        
        # -90 Deg
        T = TransMat(np.array([0, 0, 0, -np.pi/2]))
        expected_R = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ])
        np.testing.assert_array_almost_equal(T.R, expected_R)

        # 180 Deg
        T = TransMat(np.array([0, 0, 0, np.pi]))
        expected_R = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
        np.testing.assert_array_almost_equal(T.R, expected_R)

    def test_dot_product(self):
        """
        Test the tranformation order. 
        Checks whether it rotates T1 first and then T2.
        This also checks whether each TransMat rotates X first and then Z
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
        T1 = TransMat(np.array([np.pi/4, 4, 2, 0]))
        # 1. Translate x axis for 2 
        # 2. Rotates 90 degrees + Translate z axis for 4
        T2 = TransMat(np.array([np.pi/2, 4, 2, 0]))
        T3 = T1.dot(T2)

        a = 1/np.sqrt(2)
        expected_R = np.array([
            [-a, -a, 0],
            [a, -a, 0],
            [0, 0, 1]
        ])
        np.testing.assert_array_almost_equal(T3.R, expected_R)

        expected_pos = np.array([2+np.sqrt(2), np.sqrt(2), 8])
        np.testing.assert_array_almost_equal(T3.position, expected_pos)

    def test_gravity_vector(self):
        """
        1. Rotate 90 deg around the same axis (for SU)
        2. Rotate 90 deg around X axis (for SU)
        """
        g_world = np.array([0, 0, 9.81])

        Tworld2vdof = TransMat(np.array([np.pi/2, 0]))
        Tvdof2su = TransMat(np.array([0, 0, 0, np.pi/2]))

        T = Tworld2vdof.dot(Tvdof2su)
        Rworld2su = T.R.T
        g_su = np.dot(Rworld2su, g_world)

        expected_vec = np.array([0, 9.81, 0])

        np.testing.assert_array_almost_equal(g_su, expected_vec, decimal=2)

    def test_vector_rotation_with_joint_rotation(self):
        """
        1. Rotate 90 deg around the motor axis. 
        2. Rotate 90 deg around the same axis (for SU)
        3. Rotate 90 deg around X axis (for SU)
        """
        vec = np.array([1, 2, 3])

        Tjoint = TransMat(np.array(np.pi/2))
        Tjoint2vdof = TransMat(np.array([np.pi/2, 0]))
        Tvdof2su = TransMat(np.array([0, 0, 0, np.pi/2]))

        T = Tjoint.dot(Tjoint2vdof).dot(Tvdof2su)
        Rworld2su = T.R.T
        g_su = np.dot(Rworld2su, vec)

        expected_vec = np.array([-1, 3, 2])

        np.testing.assert_array_almost_equal(g_su, expected_vec, decimal=2)

    def test_panda_world_to_endeffector_su(self):
        """
        Uses Panda's DH Parameters to reach an IMU mounted on an end-effector
        Verifies if the transformation matches the expected IMU position
        """
        # DEG == 0
        joint_angles = [0, 0, 0, 0, 0, 0, 0]
        Tdofs = [TransMat(dh) for dh in PANDA_DHPARAMS]
        Tjoints = [TransMat(rad) for rad in joint_angles]
        Tdof2vdof = TransMat(np.array([np.pi/4, 0.14]))
        Tvdof2su = TransMat(np.array([0, 0.03, 0, np.pi/2]))

        # Tansform  from world to end-effector's IMU
        T = TransMat(np.zeros(4)) 
        for Tdof, Tjoint in zip(Tdofs, Tjoints):
            T = T.dot(Tdof).dot(Tjoint)
        T = T.dot(Tdof2vdof).dot(Tvdof2su)

        # Given by TF: Just run `rosrun tf tf_echo /world /imu_link6`
        expected_position = [0.125, 0.020, 0.891]
        np.testing.assert_array_almost_equal(T.position, expected_position, decimal=1)


        # DEG == 90
        joint_angles = [0, np.pi/2, 0, 0, 0, 0, 0]
        Tdofs = [TransMat(dh) for dh in PANDA_DHPARAMS]
        Tjoints = [TransMat(rad) for rad in joint_angles]
        Tdof2vdof = TransMat(np.array([np.pi/4, 0.14]))
        Tvdof2su = TransMat(np.array([0, 0.03, 0, np.pi/2]))

        # Tansform  from world to end-effector's IMU
        T = TransMat(np.zeros(4)) 
        for Tdof, Tjoint in zip(Tdofs, Tjoints):
            T = T.dot(Tdof).dot(Tjoint)
        T = T.dot(Tdof2vdof).dot(Tvdof2su)

        # Given by TF: Just run `rosrun tf tf_echo /world /imu_link6`
        expected_position = [0.557, 0.020, 0.206]
        np.testing.assert_array_almost_equal(T.position, expected_position, decimal=1)

    def test_sawyer_world_to_endeffector_su(self):
        """
        Uses Sawyer's DH Parameters to reach an IMU mounted on an end-effector
        Verifies if the transformation matches the expected IMU position
        """
        # DEG == 0
        joint_angles = [0, 0, 0, 0, 0, 0, 0]
        Tdofs = [TransMat(dh) for dh in SAWYER_DHPARAMS]
        Tjoints = [TransMat(rad) for rad in joint_angles]
        Tdof2vdof = TransMat(np.array([0, 0.1]))
        Tvdof2su = TransMat(np.array([0, 0.03, 0, -np.pi/2]))

        # Tansform  from world to end-effector's IMU
        T = TransMat(np.zeros(4)) 
        for Tdof, Tjoint in zip(Tdofs, Tjoints):
            T = T.dot(Tdof).dot(Tjoint)
        T = T.dot(Tdof2vdof).dot(Tvdof2su)

        # Given by TF: Just run `rosrun tf tf_echo /world /imu_link6`
        expected_position = [1.084, 0.131, 0.195]
        np.testing.assert_array_almost_equal(T.position, expected_position, decimal=1)

        # DEG == 90
        joint_angles = [0, -np.pi/2, 0, 0, 0, 0, 0]
        Tdofs = [TransMat(dh) for dh in SAWYER_DHPARAMS]
        Tjoints = [TransMat(rad) for rad in joint_angles]
        Tdof2vdof = TransMat(np.array([0, 0.1]))
        Tvdof2su = TransMat(np.array([0, 0.03, 0, -np.pi/2]))

        # Tansform  from world to end-effector's IMU
        T = TransMat(np.zeros(4)) 
        for Tdof, Tjoint in zip(Tdofs, Tjoints):
            T = T.dot(Tdof).dot(Tjoint)
        T = T.dot(Tdof2vdof).dot(Tvdof2su)

        # Given by TF: Just run `rosrun tf tf_echo /world /imu_link6`
        expected_position = [-0.043, 0.131, 1.319]
        np.testing.assert_array_almost_equal(T.position, expected_position, decimal=1)

class ParameterManagerTest(unittest.TestCase):
    """
    Parameter Manager Class
    """
    def test_shapes(self):
        """
        Test the shape of all lists of TransMat
        """
        poses = np.array([INIT_POSE, SECOND_POSE])
        param_manager = ParameterManager(N_JOINT, poses, BOUNDS, BOUNDS_SU)

        self.assertEqual(len(param_manager.Tdof2dof), N_JOINT)
        self.assertEqual(len(param_manager.Tdof2vdof), N_JOINT)
        self.assertEqual(len(param_manager.Tvdof2su), N_JOINT)
        self.assertEqual(len(param_manager.Tposes), 2)
        self.assertEqual(len(param_manager.Tposes[0]), N_JOINT)

    def test_get_params(self):
        """
        Test get_params function
        """
        poses = np.array([INIT_POSE, SECOND_POSE])
        param_manager = ParameterManager(N_JOINT, poses, BOUNDS, BOUNDS_SU)
        for i in range(N_JOINT):
            params, _ = param_manager.get_params_at(i=i)
            self.assertEqual(params.size, 10)
        
        param_manager = ParameterManager(N_JOINT, poses, BOUNDS, BOUNDS_SU, PANDA_DHPARAMS)
        for i in range(N_JOINT):
            params, _ = param_manager.get_params_at(i=i)
            self.assertEqual(params.size, 6)

    def test_get_tmat_until(self):
        """
        Test get_tmat_until function
        """
        poses = np.array([INIT_POSE])
        param_manager = ParameterManager(N_JOINT, poses, BOUNDS, BOUNDS_SU)

        Tdof, Tposes = param_manager.get_tmat_until(i=0)
        self.assertEqual(len(Tdof), 0)
        self.assertEqual(len(Tposes), 1)
        self.assertEqual(len(Tposes[0]), 1)

        Tdof, Tposes = param_manager.get_tmat_until(i=1)
        self.assertEqual(len(Tdof), 1)
        self.assertEqual(len(Tposes), 1)
        self.assertEqual(len(Tposes[0]), 2)

        for i in range(2, N_JOINT): 
            Tdof, Tposes = param_manager.get_tmat_until(i=i)
            self.assertEqual(len(Tdof), i)
            self.assertEqual(len(Tposes), 1)
            self.assertEqual(len(Tposes[0]), i+1)
    
    def test_set_params(self):
        """
        Test set_params function
        """
        poses = np.array([INIT_POSE])
        param_manager = ParameterManager(N_JOINT, poses, BOUNDS, BOUNDS_SU)

        raised = False
        try:
            params, _ = param_manager.get_params_at(i=0)
            param_manager.set_params_at(i=0, params=params)
        except:
            raised = True
        self.assertFalse(raised, 'Exception raised')

        raised = False
        try: 
            params, _ = param_manager.get_params_at(i=1)
            param_manager.set_params_at(i=1, params=params)
        except:
            raised = True
        self.assertFalse(raised, 'Exception raised')


if __name__ == '__main__':
    unittest.main()
