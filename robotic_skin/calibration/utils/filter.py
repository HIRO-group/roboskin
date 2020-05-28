import numpy as np


def hampel_filter_forloop(input_series, window_size, n_sigmas=3):
    """
    Implementation of Hampel Filter for outlier detection.

    Arguments
    ----------
    `input_series`: `np.array`
        The input data to use for outlier detection.

    `window_size`: `int`
        The sliding window size to use for the filter on
        `input_series`.

    `n_sigmas`: `int`
        The number of standard deviations to determine
        what data points are outliers.
    """
    n = len(input_series)
    new_series = input_series.copy()
    k = 1.4826  # scale factor for Gaussian distribution
    indices = []
    # possibly use np.nanmedian
    for i in range((window_size), (n - window_size)):
        x0 = np.median(input_series[(i - window_size):(i + window_size)])
        S0 = k * np.median(np.abs(input_series[(i - window_size):(i + window_size)] - x0))
        if (np.abs(input_series[i] - x0) > n_sigmas * S0):
            new_series[i] = x0
            indices.append(i)
    return new_series, indices


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
