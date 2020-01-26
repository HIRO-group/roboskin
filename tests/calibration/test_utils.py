import unittest
import numpy as np
from robotic_skin.calibration.utils import TransMat

class MyTransMat(unittest.TestCase):
    def test_matrix(self):
        T = TransMat(np.zeros(4))
        print('test_matrix')
        np.testing.assert_array_equal(T.mat, np.eye(4))

    def test_n_params(self):
        T = TransMat(np.random.rand(1))
        self.assertEqual(T.n_params, 1)
        
        T = TransMat(np.random.rand(2))
        self.assertEqual(T.n_params, 2)

        T = TransMat(np.random.rand(4))
        self.assertEqual(T.n_params, 4)

    def test_wrong_number_of_params(self):
        self.assertRaises(ValueError, TransMat, np.random.rand(3))

    def test_R_shape(self):
        T = TransMat(np.zeros(4))
        self.assertEqual(T.R.shape, np.zeros((3, 3)).shape)

    def test_position_shape(self):
        T = TransMat(np.zeros(4))
        self.assertEqual(T.position.shape, np.zeros(3).shape)

    def test_sub_position_into_ndarray(self):
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
        T3 = T2*T1

        a = 1/np.sqrt(2)
        expected_R = np.array([
            [-a, -a, 0],
            [a, -a, 0],
            [0, 0, 1]
        ])
        np.testing.assert_array_almost_equal(T3.R, expected_R)

        expected_pos = np.array([0, 2*np.sqrt(2), 0])
        np.testing.assert_array_almost_equal(T3.position, expected_pos)


if __name__ == '__main__':
    unittest.main()
