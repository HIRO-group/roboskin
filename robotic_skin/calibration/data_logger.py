import os
import pickle
import numpy as np
from datetime import datetime
from robotic_skin.calibration import utils


class DataLogger():
    def __init__(self, savedir, robot, method):
        self.date = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = robot + '_' + method + '.pickle'
        self.savepath = os.path.join(savedir, filepath)
        self.best_data = {}
        self.trials = {}
        self.average_euclidean_distance = 0.0

    def add_best(self, i_su, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, np.ndarray):
                value = value.tolist()

            if key not in self.best_data:
                self.best_data[key] = {}

            self.best_data[key][i_su] = value

            # Append value to np.array
            setattr(self, key, np.array(list(self.best_data[key].values())))
        self.average_euclidean_distance = np.mean(
            list(self.best_data['euclidean_distance'].values()))

    def add_trial(self, global_step, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, np.ndarray):
                value = value.tolist()

            if global_step not in self.best_data:
                self.trials[global_step] = {}

            self.trials[global_step][key] = value

    def save(self):
        data = {
            'date': self.date,
            'average_euclidean_distance': self.average_euclidean_distance,
            'best_data': self.best_data,
            'trials': self.trials}
        with open(self.savepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def print(self):
        print('Estimated SU Positions')
        for i, values in self.best_data['position'].items():
            print(f'SU{i}: {utils.n2s(np.array(values), 3)}')

        print('Estimated SU Orientations')
        for i, values in self.best_data['orientation'].items():
            print(f'SU{i}: {utils.n2s(np.array(values), 3)}')

        print('average_euclidean_distance: ', self.average_euclidean_distance)
