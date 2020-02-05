"""
Tests for Lower Traingular Matrix generation
"""
import unittest
import numpy as np
from robotic_skin.algorithm.convert_to_lowertriangular_matrix import ConvertToLT


class TestlMatrix(unittest.TestCase):
    """
    This class will just test matrix, which is convertible to Lower Triangular(LT) Matrix
    """

    def test_normal(self):
        """
        We will be inputting normal array here which is LT convertible, and check if output is LT
        Returns
        -------
        None

        """
        test_array = np.array([
            [0, 1, 0, 1, 0],
            [1, 1, 0, 1, 1],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 0, 1, 1],
            [0, 0, 0, 1, 0]
        ])
        expected_matrix = np.array([
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1]
        ])
        _, final_matrix, _, _, _ = ConvertToLT(test_array).get_lt_matrix_infos()
        np.testing.assert_array_equal(expected_matrix, final_matrix)

    def test_deformed_matrix(self):
        """
        We will pass in a deformed matrix, a condition where obtaining LT matrix is impossible. Like in the test array
        given below
        Returns
        -------
        None

        """
        # This is impossible matrix
        # In row 4, column 2, it's 0
        # Physically that means when Joint 2 is moved, All accels move except 4. Which is physically is impossible
        # As our robot is robot hand.
        # Because if Accel 2 moves, Accel 4 which is above 4 should move, unless and until you are defying the laws of
        # physics
        test_array = np.array([
            [0, 1, 0, 1, 0],
            [1, 1, 0, 1, 1],
            [0, 0, 0, 0, 0],
            [1, 0, 1, 1, 1],
            [0, 1, 0, 1, 1],
            [0, 0, 0, 1, 0]
        ])
        is_lt_matrix, final_matrix, _, _, _ = ConvertToLT(test_array).get_lt_matrix_infos()
        expected_matrix = np.array([
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 0, 1, 1]
        ])
        _, final_matrix, _, _, _ = ConvertToLT(test_array).get_lt_matrix_infos()
        np.testing.assert_array_equal(expected_matrix, final_matrix)


if __name__ == "__main__":
    unittest.main()
