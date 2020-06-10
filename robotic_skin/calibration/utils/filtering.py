import numpy as np


def low_pass_filter(data, samp_freq, cutoff_freq=15.):
    """
    Implementation of the standard pass filter,
    also known as a exponential moving average filter.

    Arguments
    ---------
    `data`:
        data to be filtered.
    `samp_freq`:
        sampling frequency of the data
    `cutoff_freq`:
        cutoff frequency; that is, data that is > = `cutoff_freq` will
        be attentuated.
    """
    # need to cut cutoff_freq in half because we apply two filters.
    half_cutoff_freq = cutoff_freq * 0.5
    n = len(data)
    # smoother data when alpha is lower
    tau = 1 / (2 * np.pi * half_cutoff_freq)
    dt = 1 / samp_freq
    alpha = dt / (dt + tau)
    new_data = data.copy()

    for i in range(1, n):
        new_data[i] = ((1 - alpha) * new_data[i-1]) + (alpha * data[i])
    reversed_data = new_data[::-1]

    for i in range(1, n):
        reversed_data[i] = ((1 - alpha) * reversed_data[i-1]) + (alpha * reversed_data[i])

    return reversed_data[::-1]
