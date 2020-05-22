import numpy as np
import pyquaternion as pyqt
from robotic_skin.calibration import utils


class Evaluator():
    def __init__(self, true_su_pose):
        self.true_su_pose = true_su_pose

    def evaluate(self, T, i_su):
        euclidean_distance = np.linalg.norm(
            T.position - self.true_su_pose[f'su{i_su+1}']['position'])

        q_su = self.true_su_pose[f'su{i_su+1}']['rotation']
        quaternion_distance = pyqt.Quaternion.absolute_distance(
            T.q, utils.np_to_pyqt(q_su))

        return {'position': euclidean_distance,
                'orientation': quaternion_distance}
