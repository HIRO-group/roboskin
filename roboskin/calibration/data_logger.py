import os
import time
import pickle
import numpy as np
from datetime import datetime
from roboskin.calibration import utils


class DataLogger():
    def __init__(self, savedir, robot, method, overwrite=False):
        self.date = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = robot + '_' + method + '.pickle'
        filepath = robot + '_' + method + '_jan16' + '.pickle'

        self.savepath = os.path.join(savedir, filepath)
        self.best_data = {}
        self.trials = {}
        self.average_euclidean_distance = 0.0
        self.start_timers = {}
        self.elapsed_times = {}
        self.overwrite = overwrite

    def start_timer(self, timer_name):
        if timer_name in self.start_timers.keys() and not self.overwrite:
            raise ValueError('Timer Name "{}" already exists'.format(timer_name))

        self.start_timers[timer_name] = time.time()

    def end_timer(self, timer_name):
        if timer_name not in self.start_timers.keys():
            raise KeyError('Timer Name "{}" does not exist'.format(timer_name))

        end_time = time.time()

        elapsed_time = end_time - self.start_timers[timer_name]
        self.elapsed_times[timer_name] = elapsed_time

        return elapsed_time

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

    def add_trial(self, global_step, imu_num, **kwargs):
        if global_step not in self.best_data:
            self.trials[global_step] = {}
        self.trials[global_step]['imu_num'] = imu_num
        for key, value in kwargs.items():
            if isinstance(value, np.ndarray):
                value = value.tolist()

            self.trials[global_step][key] = value

    def save(self):
        data = {
            'date': self.date,
            'average_euclidean_distance': self.average_euclidean_distance,
            'best_data': self.best_data,
            'trials': self.trials}
        with open(self.savepath, 'wb') as f:
            pickle.dump(data, f, protocol=2)

    def __call__(self):
        print('Estimated SU Positions')
        for i, values in self.best_data['position'].items():
            print('SU{}: {}'.format(i, utils.n2s(np.array(values), 3)))

        print('Estimated SU Orientations')
        for i, values in self.best_data['orientation'].items():
            print('SU{}: {}'.format(i, utils.n2s(np.array(values), 3)))

        print('average_euclidean_distance: ', self.average_euclidean_distance)
