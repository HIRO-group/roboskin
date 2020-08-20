"""
Testing utils module
"""
import unittest
import numpy as np
from pyquaternion import Quaternion
from roboskin.calibration import utils


class QuaternionTest(unittest.TestCase):
    def test_quaternion_l2_distance(self):
        """
        Tests quaternion distance function.
        """
        q1 = Quaternion(axis=[1, 0, 0], angle=np.pi/2)
        q2 = Quaternion(axis=[0, 1, 0], angle=np.pi/2)

        error = utils.quaternion_l2_distance(q1, q2)
        self.assertAlmostEqual(error, 1)

        q1 = Quaternion(axis=[1, 0, 0], angle=np.pi/2)
        q2 = Quaternion(axis=[1, 0, 0], angle=-np.pi/2)

        error = utils.quaternion_l2_distance(q1, q2)
        self.assertAlmostEqual(error, 2)

    def test_quaternion_from_two_vectors(self):
        """
        Tests getting a quaternion from
        two different vectors.
        """
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])
        q = utils.quaternion_from_two_vectors(source=v1, target=v2)

        assert q == Quaternion(axis=[0, 0, 1], angle=np.pi/2)

    def test_quaternion_angles(self):
        a = Quaternion(axis=[1, 0, 0], angle=-np.pi / 2)
        b = Quaternion(axis=[1, 0, 0], angle=np.pi / 3)
        a = np.array([a[1], a[2], a[3], a[0]])
        b = np.array([b[1], b[2], b[3], b[0]])
        angle_in_degrees = utils.angle_between_quaternions(a, b)
        self.assertAlmostEqual(angle_in_degrees, 2.6179938779914944)


if __name__ == '__main__':
    unittest.main()
