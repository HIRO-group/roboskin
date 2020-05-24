import time
import numpy as np
import pyquaternion as pyqt
from robotic_skin.calibration import utils


class Evaluator():
    def __init__(self, true_su_pose):
        self.true_su_pose = true_su_pose
        self.start_timers = {}

    def evaluate(self, T, i_su):
        euclidean_distance = np.linalg.norm(
            T.position - self.true_su_pose[f'su{i_su+1}']['position'])

        q_su = self.true_su_pose[f'su{i_su+1}']['rotation']
        quaternion_distance = pyqt.Quaternion.absolute_distance(
            T.q, utils.np_to_pyqt(q_su))

        return {'position': euclidean_distance,
                'orientation': quaternion_distance}

    def start_timer(self, timer_name=None):
        if timer_name is None:
            timer_name = 'global'

        if timer_name in self.start_timers.keys():
            raise ValueError(f'Timer Name: {timer_name} already exists')

        self.start_timers[timer_name] = time.time()

    def end_timer(self, timer_name=None):
        if timer_name is None:
            timer_name = 'global'

        if timer_name not in self.start_timers.keys():
            raise KeyError(f'Timer Name: {timer_name} does not exist')

        end_time = time.time()

        return end_time - self.start_timers[timer_name]
