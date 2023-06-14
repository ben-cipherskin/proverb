from scipy import signal
from scipy.signal import detrend
from scipy.fft import fft, fftfreq
import scipy.interpolate as interp
import numpy as np
import pandas as pd
import datetime as dt


def butter_lowpass(sig, highcut, fs, order):
    nyq = 0.5 * fs
    high = highcut / nyq
    sos = signal.butter(order, high, btype='low', output='sos')
    return signal.sosfilt(sos, sig)


def interpolate_to_hz(times, y, fs=1, detrend_signal=False, order=1):
    """
    interpolates and resamples y to a new sample frequency (fs) in Hz by using the current time stamps.
    :param times: array in containing the time information (in seconds!) of the signal
    :param y: actual signal
    :param fs: frequency to interpolate too
    :param detrend_signal: boolean on whether or not to detrend signal
    :param order: An int of the type of order of the signal
    :return: new timestamps with associated y values (both arrays)
    """
    # new_times = [t/fs for t in range(t_min*fs, t_max*fs)]  # range only does integers, so adjusted for that
    new_times = np.arange(times[0], times[-1], 1 / fs)
    new_times = np.around(new_times, 3)
    interpolation = interp.interp1d(times, y, fill_value='extrapolate')
    new_y = interpolation(new_times)
    if detrend_signal:
        if order == 1:
            det_type = 'constant'
        elif order == 2:
            det_type = 'linear'
        else:
            print('wrong order of detrending')

        new_y = detrend(new_y, type=det_type, overwrite_data=True)
    return new_times, new_y


def time_to_seconds(times):
    """
    :param times: Series or List; the time information stored as datetimes or ms
    :return: seconds: List; The same timestamp values as times converted to seconds
    """
    if type(times) == list:
        if isinstance(times[0], dt.date):
            times = pd.to_datetime(times)
            seconds = [(s - times[0]).total_seconds() for s in times]
        else:
            seconds = [(t - times[0]) / 1000 for t in times]
    elif pd.api.types.is_int64_dtype(times) or pd.api.types.is_float_dtype(times):  # Assumed to be ms already
        times.reset_index(drop=True, inplace=True)
        seconds = [(t - times[0]) / 1000 for t in times]
    else:
        print(f"dtype {times.dtype} not recognized of {times}")
    return seconds


def fft_calc(section, fs, pad_len):
    """
    :param section:
    :param fs:
    :param pad_len:
    :return:
    """
    N = len(section)
    T = 1 / fs
    n = N + pad_len
    yf = fft(section, n=n)
    y = 2.0 / N * np.abs(yf[0:N // 2])
    xf = fftfreq(N, T)[:N // 2]
    return xf, y


def calc_dom_freq_general(t_signal, signal, interpolation_freq=50, start_idx_lag=150):
    """

    :param t_signal
    :param signal:
    :param interpolation_freq:
    :param start_idx_lag:
    :param plot:
    :param save_fig:
    :param title_add:
    :return:
    """
    fs = interpolation_freq
    plotting_data = {}
    new_times, new_section = interpolate_to_hz(t_signal, signal, fs=fs, detrend_signal=True, order=2)
    xf, y = fft_calc(new_section, fs, pad_len=0)
    dom_x = np.argmax(y[start_idx_lag:]) + start_idx_lag
    plotting_data["Interpolated Times"] = new_times
    plotting_data["Interpolated Signal"] = new_section
    plotting_data["xf"] = xf
    plotting_data["y"] = y
    plotting_data["Dom X"] = dom_x
    return plotting_data
