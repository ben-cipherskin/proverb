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


def transform_global_t(global_t):
    """
    :param global_t: array of strings containing time information as coming from the sleeve.
    :return: array of floats with time information in seconds with first one being 0.
    """
    zero_t = dt.datetime.strptime(global_t[0].replace("T", " ").replace("Z", ""), "%Y-%m-%d %H:%M:%S.%f")
    new_times = []
    # Calculate time from start time in minutes
    for t in global_t:
        try:
            new_times.append((dt.datetime.strptime(t.replace("T", " ").replace("Z", ""), "%Y-%m-%d %H:%M:%S.%f") - zero_t).total_seconds())
        except ValueError:
            new_times.append((dt.datetime.strptime(t.replace("T", " ").replace("Z", ""), "%Y-%m-%d %H:%M:%S") - zero_t).total_seconds())
    return new_times


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


def filter_3d_signal(time_series, signals, order=2, highcut=100, lowcut=0, bandpass=False, highpass=False):
    """
    Function to filter 3d signal using a butterworth filter as implemented in scipy. The filter is applied to each axis
    of the signal separately and then combined again and returned.
    :param time_series: list; time information of the signal
    :param signals: list of lists; the 3d signal
    :param order: int; the order of the butterworth filter
    :param highcut: float; the highcut frequency in Hz
    :param lowcut: float; the lowcut frequency in Hz
    :param bandpass: bool; whether or not to apply a bandpass filter
    :param highpass: bool; whether or not to apply a highpass filter
    :return time_series: list; time information of the signal
    :return final_signals: list of lists; the filtered 3d signal
    """
    if bandpass:
        if highpass:
            print("Both bandpass and highpass are True. Please choose only one.")
            return
        else:
            print("Filtering with bandpass filter.")
    elif highpass:
        print("Filtering with highpass filter.")
    else:
        print("No filtering applied.")
        return time_series, signals

    # Convert to numpy array
    signals = np.array(signals)
    # Get sampling frequency
    fs = 1 / (time_series[1] - time_series[0])
    # Get nyquist frequency
    nyq = 0.5 * fs
    # Get cutoff frequencies
    high = highcut / nyq
    low = lowcut / nyq
    # Get filter order
    order = 2
    # Get filter type
    if bandpass:
        btype = "band"
    elif highpass:
        btype = "high"
    else:
        btype = "low"
    # Get filter coefficients
    sos = signal.butter(order, [low, high], btype=btype, output='sos')
    # Apply filter to each axis
    filtered_signals = signal.sosfilt(sos, signals, axis=0)
    # Convert back to list
    final_signals = filtered_signals.tolist()
    return time_series, final_signals

