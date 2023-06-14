import statistics
import numpy as np
import pandas as pd
import math
from scipy.signal import find_peaks
from dsp import butter_lowpass, interpolate_to_hz, calc_dom_freq_general
from CONSTANTS import lap_distance


def detect_dist_from_sig(sig_list, sig_type, lap_distance, height=0.1, distance=50, prominence=0.1):
    """
    Use a peak finder to calculate total distance covered
    :param prominence: float; parameter for peak finder
    :param distance: float; parameter for peak finder
    :param height: float; parameter for peak finder
    :param sig_list: lst; single quaternion value (i.e. w only or z only)
    :param subject: str; subject name
    :param date: str; date of experiment
    :param sig_type: str; which part of quaternion/euler/angle used
    :return total_dist: float; the total distance in meters covered in the walk test
    :return sorted_turns: lst; list of peaks and troughs which represents each turn at a cone
    """

    sig_peaks, _ = find_peaks(sig_list, width=None, height=height, distance=distance, prominence=prominence)
    # Calculate distances of peaks
    peak_dist = list(np.diff(sig_peaks))
    peak_dist.insert(0, sig_peaks[0])
    peak_mean_dist = statistics.mean(peak_dist)
    new_distance = statistics.median(peak_dist) * 0.9
    sig_peaks, _ = find_peaks(sig_list, width=None, height=height, distance=new_distance, prominence=prominence)
    if sig_peaks[0] < 0.75 * peak_mean_dist:
        sig_peaks = [val for val in sig_peaks if val != sig_peaks[0]]

    # print(f'Num Turns Detected: {len(sig_peaks)}')
    # Sort the peak list
    sorted_turns = sorted(list(sig_peaks))
    # sorted_turns = sorted(list(sig_troughs) + list(sig_peaks))
    # This calculation is using the assumption that the subject maintains a constant speed throughout the 6 minutes
    avg_samples_per_turn_w = sum(np.diff(sorted_turns)) / (len(sorted_turns) - 1)
    last_turn_idx = sorted_turns[-1]
    # Calculate starting and final distances using the distance between the cones
    start_dist = (len(sig_peaks)) * lap_distance
    final_turn_len = len(sig_list[last_turn_idx:])
    final_turn_dist = round(final_turn_len / avg_samples_per_turn_w * lap_distance, 2)
    total_dist = start_dist + final_turn_dist
    # print(f'Total Distance {sig_type}: {total_dist} meters')
    return total_dist, sorted_turns


def estimate_total_distance_gyro(gyro_rms, gyros, plot=False, save_fig=False):
    """

    :param gyro_rms: lst; list of prox ([0]) and dist ([1]) rms from the gyroscope
    :param plot_info: lst; list containing information for naming plots
    :param save_fig: bool; whether to save generated figure
    :return:
    """
    # Sum each index for the RMS where the index exists for both lists
    # (i.e. resultant combined rms is length of shorter list)
    combined_rms = [sum(i) for i in zip(gyro_rms[1], gyro_rms[3])]
    prox_gyro_t = gyro_rms[0]
    prox_rms = (butter_lowpass(gyro_rms[1], highcut=0.20, fs=100, order=2))
    total_distance_prox, sorted_turns_prox = detect_dist_from_sig(prox_rms, 'Prox Gyro RMS', lap_distance, height=0.1,
                                                                  distance=100, prominence=0.1)
    dist_gyro_t = gyro_rms[2]
    dist_rms = (butter_lowpass(gyro_rms[3], highcut=0.20, fs=100, order=2))
    total_distance_dist, sorted_turns_dist = detect_dist_from_sig(dist_rms, 'Dist Gyro RMS', lap_distance, height=0.1,
                                                                  distance=100, prominence=0.1)

    gyro_labels = ["X", "Y", "Z"]
    gyro_colors = ["blue", "red", "green"]
    gyros_unfiltered_prox_t = [(val - gyros[0][0]) / 1000 for val in gyros[0]]
    gyros_unfiltered_dist_t = [(val - gyros[2][0]) / 1000 for val in gyros[2]]
    _, final_gyro_idx_prox = find_nearest(gyros_unfiltered_prox_t, prox_gyro_t[sorted_turns_prox[-1]])
    _, final_gyro_idx_dist = find_nearest(gyros_unfiltered_dist_t, dist_gyro_t[sorted_turns_dist[-1]])

    return total_distance_prox, total_distance_dist, sorted_turns_prox, sorted_turns_dist


def split_laps(acc_dist_t, dist_x, dist_y, dist_z, gyro_dist_t, sorted_turns_dist):
    acc = [[dist_x[i], dist_y[i], dist_z[i]] for i in range(len(dist_x))]
    start_idx, prev_time = 0, 0
    steps_in_each_turn = []
    freq_in_each_turn = []
    time_in_each_turn = []
    for idx, time in enumerate(np.asarray(gyro_dist_t)[sorted_turns_dist]):
        _, gyro_idx = find_nearest(acc_dist_t, time)
        temp_sig_t = acc_dist_t[start_idx:gyro_idx]
        temp_sig = acc[start_idx:gyro_idx]
        num_steps, max_freq = step_detection_algo(temp_sig_t, temp_sig, counter=idx)
        steps_in_each_turn.append(num_steps)
        freq_in_each_turn.append(max_freq)
        time_in_each_turn.append(time - prev_time)
        prev_time = time
        start_idx = gyro_idx
    temp_sig_t = acc_dist_t[gyro_idx:]
    temp_sig = acc[gyro_idx:]
    final_lap_steps, final_freq = step_detection_algo(temp_sig_t, temp_sig, counter=idx + 1)
    freq_in_each_turn.append(final_freq)
    time_to_add = 3  # Time in seconds to add to the last lap
    time_in_each_turn.append(acc_dist_t[-1] + time_to_add - prev_time)  # Add 5 seconds to account for time loss
    avg_steps = sum(steps_in_each_turn) / len(steps_in_each_turn)
    # Add last lap to steps_in_each_turn arr after avg is calculated
    steps_in_each_turn.append(final_lap_steps)
    # print(f'Steps in Each Turn: {steps_in_each_turn}')
    return steps_in_each_turn, avg_steps, final_lap_steps, freq_in_each_turn, time_in_each_turn


def filtered_dc_blocked_norm(sig_t, sig):
    """
    Calculate the norm (RMS) of a signal and then remove the rolling average (size 20) from it (DC value) to center around zero
    :param sig_t:
    :param sig:
    :return:
    """
    rms = [0] * len(sig)
    for i in range(len(sig)):
        sums = 0
        for j in range(len(sig[0])):
            sums += sig[i][j] ** 2
        rms[i] = math.sqrt(sums)
    rms_df = pd.DataFrame(rms)
    rms_rolling = rms_df.rolling(20).mean()
    dc_block_norm = [val - rms_rolling.iloc[idx + 19] for idx, val in enumerate(rms[20:])]
    interp_t, interp_rms = interpolate_to_hz(sig_t[:-20], [val[0] for val in dc_block_norm], fs=100,
                                             detrend_signal=False)
    filt_signal = butter_lowpass(interp_rms, highcut=20, fs=100, order=2)
    return interp_t, filt_signal


def step_detection_algo(sig_t, sig, dc_norm_filt=True, counter=0):
    """
    Implementation of step counting algorithm: Design and Implementation of Practical Step
    Detection Algorithm for Wrist-Worn Devices. DOI:10.1109/JSEN.2016.2603163
    :param sig_t:
    :param sig:
    :param counter:
    :return:
    """
    if type(counter) == str:
        new_counter = counter.split(' ')[0] + ' ' + str(int(counter.split(' ')[1]) + 1)
    else:
        new_counter = counter + 1
    sig_t = [val - sig_t[0] for val in sig_t]
    if dc_norm_filt:
        sig_t, filt_signal = filtered_dc_blocked_norm(sig_t, sig)
        plotting_bool = False
        avg_thrsh = [val for val in pd.DataFrame(filt_signal).rolling(10).mean()[0]]
        bit_stream = [-1] * len(filt_signal)
        # Convert signal to bit stream (binary of positive or negative slope)
        for i in range(1, len(filt_signal)):
            if filt_signal[i] > filt_signal[i - 1]:
                bit_stream[i] = 1
            else:
                bit_stream[i] = -1
        idxs_min, idxs_max = [], []
        # Identify all local min and max's
        for idx, val in enumerate(bit_stream[:-1]):
            if val == -1 and bit_stream[idx + 1] == 1:
                idxs_min.append(idx)
            elif val == 1 and bit_stream[idx + 1] == -1:
                idxs_max.append(idx)
        # Check first condition to find true local min and max's (the max is above the avg threshold and the min is below)
        idx_to_remove = []
        for i, idx in enumerate(idxs_max):
            if filt_signal[idx] < avg_thrsh[idx]:
                idx_to_remove.append(i)
        idxs_max = [val for i, val in enumerate(idxs_max) if i not in idx_to_remove]
        idx_to_remove = []
        for i, idx in enumerate(idxs_min):
            if filt_signal[idx] > avg_thrsh[idx]:
                idx_to_remove.append(i)
        idxs_min = [val for i, val in enumerate(idxs_min) if i not in idx_to_remove]
        # Check second condition to find true local min and max's (A max is preceded by a min and vice versa)
        min_start, max_start = False, False
        min_idx_to_keep, max_idx_to_keep = [], []
        if idxs_min[0] < idxs_max[0]:
            min_start = True
            min_idx_to_keep.append(0)
        else:
            max_start = True
            max_idx_to_keep.append(0)
        last_val = -1
        previous_min_idx, previous_max_idx = 0, 0
        while True:
            if last_val >= idxs_max[-1] or last_val >= idxs_min[-1]:
                break
            if min_start:
                last_val = idxs_min[min_idx_to_keep[-1]]
                for i, val in enumerate(idxs_max[previous_max_idx:]):
                    if val > last_val:
                        max_idx_to_keep.append(i + previous_max_idx)
                        last_val = val
                        previous_max_idx += i
                        break
                for i, val in enumerate(idxs_min[previous_min_idx:]):
                    if val > last_val:
                        min_idx_to_keep.append(i + previous_min_idx)
                        last_val = val
                        previous_min_idx += i
                        break
            if max_start:
                last_val = idxs_max[max_idx_to_keep[-1]]
                for i, val in enumerate(idxs_min[previous_min_idx:]):
                    if val > last_val:
                        min_idx_to_keep.append(i + previous_min_idx)
                        last_val = val
                        previous_min_idx += i
                        break
                for i, val in enumerate(idxs_max[previous_max_idx:]):
                    if val > last_val:
                        max_idx_to_keep.append(i + previous_max_idx)
                        last_val = val
                        previous_max_idx += i
                        break
        idxs_min = [val for i, val in enumerate(idxs_min) if i in min_idx_to_keep]
        idxs_max = [val for i, val in enumerate(idxs_max) if i in max_idx_to_keep]
    else:
        # Classic Peak finding instead of step detection algo
        plotting_bool = False
        sig_t, interp_sig = interpolate_to_hz(sig_t, sig, fs=100, detrend_signal=False)
        filt_signal = butter_bandpass(interp_sig, lowcut=0.5, highcut=2.5, fs=100, order=2)

        idxs_max, _ = find_peaks(filt_signal, height=0.8, prominence=1, distance=25)
        idxs_min, _ = find_peaks([-1 * val for val in filt_signal], height=0.8, prominence=1, distance=25)

    plotting_bool = False  # plotting bool override
    fft_plotting_data = calc_dom_freq_general(t_signal=sig_t, signal=filt_signal, start_idx_lag=0,
                                              interpolation_freq=100)
    # max_freq_of_section = fft_plotting_data['xf'][fft_plotting_data["Dom X"]]
    freq_cutoff = 2.5
    max_freq_of_section = fft_plotting_data["xf"][
        np.argmax(fft_plotting_data['y'][:find_nearest(fft_plotting_data['xf'], freq_cutoff)[1]])]
    return min(len(idxs_min), len(idxs_max)), max_freq_of_section


def find_nearest(array, value):
    """
    Function that returns the nearest value to the desired value as well as its index in the array that is sent into the function
    :param array: the array which will be used to find the relevant value within
    :param value: the value to look for within the array
    :return array[idx]: the value found within the array that is the nearest in location to the input value
    :return idx:  the index of the found value within the input array
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx
