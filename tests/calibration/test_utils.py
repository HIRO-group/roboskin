import unittest
import numpy as np
from robotic_skin.calibration.utils import TransMat

class MyTransMat(unittest.TestCase):
    def test_matrix(self):
        T = TransMat(np.zeros(4))
        print('test_matrix')
        np.testing.assert_array_equal(T.mat, np.eye(4))

if __name__ == '__main__':
    unittest.main()
