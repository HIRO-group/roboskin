"""
Testing utils module
"""
import unittest
import numpy as np
from robotic_skin.calibration.utils import TransMat, ParameterManager

N_JOINT = 7
INIT_POSE = np.zeros(N_JOINT)
BOUNDS = np.array([
    [-np.pi, 0.0, -0.1, -np.pi],
    [np.pi, 0.5, 0.1, np.pi]
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
        print('test_matrix')
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

    def test_tmat_90degrees(self):
        """
        Test a tranformation matrix by rotating 90 degrees
        """
        T = TransMat(np.array([np.pi/2, 0, 2, 0]))
        expected_R = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        np.testing.assert_array_almost_equal(T.R, expected_R)

        expected_pos = np.array([0, 2, 0])
        np.testing.assert_array_almost_equal(T.position, expected_pos)
    
    def test_tmat_45degrees(self):
        """
        Test a tranformation matrix by rotating 45 degrees
        """
        T = TransMat(np.array([np.pi/4, 0, 2, 0]))
        a = 1/np.sqrt(2)
        expected_R = np.array([
            [a, -a, 0],
            [a, a, 0],
            [0, 0, 1]
        ])
        np.testing.assert_array_almost_equal(T.R, expected_R)

        expected_pos = np.array([np.sqrt(2), np.sqrt(2), 0])
        np.testing.assert_array_almost_equal(T.position, expected_pos)

    def test_dot_product(self):
        """
        Test the tranformation order. 
        Checks wheter it rotates T1 first and then T2.
        It should look like
            ^
            | \ 
            |   \  90 degrees
            |   /
            | / 45 degrees
            -------->
        The resulting position should be [0, 2*sqrt(2)]
        """
        # Rotates 45 degrees first
        T1 = TransMat(np.array([np.pi/4, 0, 2, 0]))
        # Then 90 eegress
        T2 = TransMat(np.array([np.pi/2, 0, 2, 0]))
        T3 = T1.dot(T2)

        a = 1/np.sqrt(2)
        expected_R = np.array([
            [-a, -a, 0],
            [a, -a, 0],
            [0, 0, 1]
        ])
        np.testing.assert_array_almost_equal(T3.R, expected_R)

        expected_pos = np.array([0, 2*np.sqrt(2), 0])
        np.testing.assert_array_almost_equal(T3.position, expected_pos)

    def test_gravity_vector(self):
        g_world = np.array([0, 0, 9.81])

        T_world2vdof = TransMat(np.array([1.57, 0.0, 0.0, 1.57]))
        T_vdof2su = TransMat(np.array([-1.57, 0.00]))

        T = T_world2vdof.dot(T_vdof2su)
        R_world2su = T.R.T
        g_su = np.dot(R_world2su, g_world)

        expected_vec = np.array([-9.81, 0, 0])

        np.testing.assert_array_almost_equal(g_su, expected_vec, decimal=2)

    def test_gravity_vector_with_joint_rotation(self):
        vec = np.array([1, 2, 3])

        T_joint = TransMat(np.array(np.pi/2))
        T_joint2vdof = TransMat(np.array([np.pi/2, 0.0, 0.0, np.pi/2]))
        T_vdof2su = TransMat(np.array([-np.pi/2, 0.00]))

        T = T_joint.dot(T_joint2vdof).dot(T_vdof2su)
        R_world2su = T.R.T
        g_su = np.dot(R_world2su, vec)

        expected_vec = np.array([-3, -1, 2])

        np.testing.assert_array_almost_equal(g_su, expected_vec, decimal=2)

class ParameterManagerTest(unittest.TestCase):
    """
    Parameter Manager Class
    """
    def test_shapes(self):
        """
        Test the shape of all lists of TransMat
        """
        poses = np.array([INIT_POSE])
        param_manager = ParameterManager(N_JOINT, poses, BOUNDS)

        self.assertEqual(len(param_manager.Tdof2dof), N_JOINT-1)
        self.assertEqual(len(param_manager.Tdof2vdof), N_JOINT)
        self.assertEqual(len(param_manager.Tvdof2su), N_JOINT)
        self.assertEqual(len(param_manager.Tposes), 1)
        self.assertEqual(len(param_manager.Tposes[0]), N_JOINT)

    def test_get_params(self):
        """
        Test get_params function
        """
        poses = np.array([INIT_POSE])
        param_manager = ParameterManager(N_JOINT, poses, BOUNDS)

        params, _ = param_manager.get_params_at(i=0)
        print(params)
        self.assertEqual(params.size, 6)
        
        for i in range(1, N_JOINT):
            params, _ = param_manager.get_params_at(i=i)
            self.assertEqual(params.size, 10)

    def test_get_tmat_until(self):
        """
        Test get_tmat_until function
        """
        poses = np.array([INIT_POSE])
        param_manager = ParameterManager(N_JOINT, poses, BOUNDS)

        Tdof, Tposes = param_manager.get_tmat_until(i=0)
        self.assertEqual(len(Tdof), 0)
        self.assertEqual(len(Tposes), 1)
        self.assertEqual(len(Tposes[0]), 1)

        Tdof, Tposes = param_manager.get_tmat_until(i=1)
        self.assertEqual(len(Tdof), 0)
        self.assertEqual(len(Tposes), 1)
        self.assertEqual(len(Tposes[0]), 2)

        for i in range(2, N_JOINT): 
            Tdof, Tposes = param_manager.get_tmat_until(i=i)
            self.assertEqual(len(Tdof), i-1)
            self.assertEqual(len(Tposes), 1)
            self.assertEqual(len(Tposes[0]), i+1)
    
    def test_set_params(self):
        """
        Test set_params function
        """
        poses = np.array([INIT_POSE])
        param_manager = ParameterManager(N_JOINT, poses, BOUNDS)

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
