"""
Testing acceleration position module.
"""
import unittest
import numpy as np
from robotic_skin.calibration.accel_position import ParameterManager
# from robotic_skin.calibration.accel_position import ParameterManager, KinematicEstimator, collect_data

N_JOINT = 7
INIT_POSE = np.zeros(N_JOINT)
BOUNDS = np.array([
    [-np.pi, 0.0, -0.1, -np.pi],
    [np.pi, 0.5, 0.1, np.pi]
    ])

class KinematicEstimatorTest(unittest.TestCase):
    """
    Tests for Kinematics Estimator.
    """
    def test_initialization(self):
        """
        tests the initialization of the KinematicEstimator.
        """
        #data, poses = collect_data()
        #estimator = KinematicEstimator(data, poses)
        #estimator.optimize()

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
        self.assertEqual(len(Tdof), 1)
        self.assertEqual(len(Tposes), 1)
        self.assertEqual(len(Tposes[0]), 1)

        Tdof, Tposes = param_manager.get_tmat_until(i=1)
        self.assertEqual(len(Tdof), 1)
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
