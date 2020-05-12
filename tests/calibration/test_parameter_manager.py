"""
Testing utils module
"""
import unittest
import numpy as np
from robotic_skin.calibration.parameter_manager import ParameterManager

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


class ParameterManagerTest(unittest.TestCase):
    """
    Parameter Manager Class
    """
    def test_shapes(self):
        """
        Test the shape of all lists of TransformationMatrix
        """
        param_manager = ParameterManager(N_JOINT, BOUNDS, BOUNDS_SU)

        self.assertEqual(len(param_manager.Tdof2dof), N_JOINT)
        self.assertEqual(len(param_manager.Tdof2vdof), N_JOINT)
        self.assertEqual(len(param_manager.Tvdof2su), N_JOINT)

    def test_get_params(self):
        """
        Test get_params function
        """
        param_manager = ParameterManager(N_JOINT, BOUNDS, BOUNDS_SU)
        for i in range(N_JOINT):
            params, _ = param_manager.get_params_at(i=i)
            self.assertEqual(params.size, 10)

        param_manager = ParameterManager(N_JOINT, BOUNDS, BOUNDS_SU, PANDA_DHPARAMS)
        for i in range(N_JOINT):
            params, _ = param_manager.get_params_at(i=i)
            self.assertEqual(params.size, 6)

    def test_get_tmat_until(self):
        """
        Test get_tmat_until function
        """
        param_manager = ParameterManager(N_JOINT, BOUNDS, BOUNDS_SU)

        for i in range(0, N_JOINT):
            Tdof = param_manager.get_tmat_until(i=i)
            self.assertEqual(len(Tdof), i+1)

    def test_set_params(self):
        """
        Test set_params function
        """
        param_manager = ParameterManager(N_JOINT, BOUNDS, BOUNDS_SU)

        raised = False
        try:
            params, _ = param_manager.get_params_at(i=0)
            param_manager.set_params_at(i=0, params=params)
        except Exception:
            raised = True
        self.assertFalse(raised, 'Exception raised')

        raised = False
        try:
            params, _ = param_manager.get_params_at(i=1)
            param_manager.set_params_at(i=1, params=params)
        except Exception:
            raised = True
        self.assertFalse(raised, 'Exception raised')


if __name__ == '__main__':
    unittest.main()
