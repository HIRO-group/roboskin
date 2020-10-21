import numpy as np
import torch
import pyquaternion as pyqt
from roboskin.calibration import utils


class Evaluator():
    def __init__(self, true_su_pose):
        self.true_su_pose = true_su_pose

    def evaluate(self, T, i_su):
        position = T.position
        if type(position) == torch.Tensor:
            position = position.cpu().detach().numpy()
        euclidean_distance = np.linalg.norm(
            position - self.true_su_pose['su{}'.format(i_su+1)]['position'])

        q_su = self.true_su_pose['su{}'.format(i_su+1)]['rotation']
        quaternion_distance = pyqt.Quaternion.absolute_distance(
            T.q, utils.np_to_pyqt(q_su))

        return {'position': euclidean_distance,
                'orientation': quaternion_distance}
