import numpy as np
import math
import os
import glob
import scipy.interpolate as interp
from scipy.signal import detrend
from scipy import signal
from pyquaternion import Quaternion
from dsp import time_to_seconds, interpolate_to_hz, butter_lowpass


def get_gyro(data, sleeve_num=0):
    """
    Gyroscope data saved in json as rotations about the x, y, z acceleration axes without orientation
    :param data:
    :param sleeve_num:
    :return:
    """
    try:
        data['rawData']['bgxData'][sleeve_num]['Gyroscope Data']
    except KeyError:
        print('No Gyroscope Data Present in this Recording')
        return []
    prox_gyro = [0] * len(data['rawData']['bgxData'][sleeve_num]['Gyroscope Data'][0]['data'])
    dist_gyro = [0] * len(data['rawData']['bgxData'][sleeve_num]['Gyroscope Data'][1]['data'])
    prox_gyro_t = [0] * len(data['rawData']['bgxData'][sleeve_num]['Gyroscope Data'][0]['data'])
    dist_gyro_t = [0] * len(data['rawData']['bgxData'][sleeve_num]['Gyroscope Data'][1]['data'])
    for i in range(len(prox_gyro)):
        temp = data['rawData']['bgxData'][sleeve_num]['Gyroscope Data'][0]['data'][i]
        prox_gyro[i] = [val / 2 ** 9 for val in temp['value']]  # Each value is divided by 2^9 because the data coming
        # from the sensor has a Q point of 9
        prox_gyro_t[i] = temp['sleeve local timestamp']
    for i in range(len(dist_gyro)):
        temp = data['rawData']['bgxData'][sleeve_num]['Gyroscope Data'][1]['data'][i]
        dist_gyro[i] = [val / 2 ** 9 for val in temp['value']]
        dist_gyro_t[i] = temp['sleeve local timestamp']
    return prox_gyro_t, prox_gyro, dist_gyro_t, dist_gyro


def get_gyro_rms(data, filt=False):
    """

    :param filt:
    :param data:
    :return:
    """
    gyros = get_gyro(data, sleeve_num=0)
    prox_gyro_t, prox_gyro, dist_gyro_t, dist_gyro = gyros[0], gyros[1], gyros[2], gyros[3]

    # Extracting quaternion components
    gyroprox_x = [val[0] for val in prox_gyro]
    gyroprox_y = [val[1] for val in prox_gyro]
    gyroprox_z = [val[2] for val in prox_gyro]

    gyrodist_x = [val[0] for val in dist_gyro]
    gyrodist_y = [val[1] for val in dist_gyro]
    gyrodist_z = [val[2] for val in dist_gyro]

    # Putting data in dictionary for plotting purposes
    gyro_prox_times = time_to_seconds(prox_gyro_t)
    gyro_dist_times = time_to_seconds(dist_gyro_t)

    if filt:
        highcut = 0.15  # Hz
        fs = 100  # Hz
        order = 2
        # Error handling for cases where timestamps have multiple zero starts
        if gyro_prox_times[1] == 0.0:
            gyro_prox_times[1] = 0.01
        gyroprox_xt, gyroprox_x = interpolate_to_hz(gyro_prox_times, gyroprox_x, fs, detrend_signal=True, order=1)
        gyroprox_yt, gyroprox_y = interpolate_to_hz(gyro_prox_times, gyroprox_y, fs, detrend_signal=True, order=1)
        gyro_prox_times, gyroprox_z = interpolate_to_hz(gyro_prox_times, gyroprox_z, fs, detrend_signal=True, order=1)
        # Error handling for cases where timestamps have multiple zero starts
        if gyro_dist_times[1] == 0.0:
            gyro_dist_times[1] = 0.01
        gyrodist_xt, gyrodist_x = interpolate_to_hz(gyro_dist_times, gyrodist_x, fs, detrend_signal=True, order=1)
        gyrodist_yt, gyrodist_y = interpolate_to_hz(gyro_dist_times, gyrodist_y, fs, detrend_signal=True, order=1)
        gyro_dist_times, gyrodist_z = interpolate_to_hz(gyro_dist_times, gyrodist_z, fs, detrend_signal=True, order=1)

        # Applying lowpass filter to quaternion components
        gyroprox_x = (butter_lowpass(gyroprox_x, highcut=highcut, fs=fs, order=order))
        gyroprox_y = (butter_lowpass(gyroprox_y, highcut=highcut, fs=fs, order=order))
        gyroprox_z = (butter_lowpass(gyroprox_z, highcut=highcut, fs=fs, order=order))

        gyrodist_x = (butter_lowpass(gyrodist_x, highcut=highcut, fs=fs, order=order))
        gyrodist_y = (butter_lowpass(gyrodist_y, highcut=highcut, fs=fs, order=order))
        gyrodist_z = (butter_lowpass(gyrodist_z, highcut=highcut, fs=fs, order=order))

    rms_prox = []
    rms_dist = []
    for j in range(len(gyroprox_x)):
        rms_prox.append(math.sqrt(((gyroprox_x[j] ** 2) + (gyroprox_y[j] ** 2) +
                                   (gyroprox_z[j] ** 2)) / 3))
    for j in range(len(gyrodist_x)):
        rms_dist.append(math.sqrt(((gyrodist_x[j] ** 2) + (gyrodist_y[j] ** 2) +
                                   (gyrodist_z[j] ** 2)) / 3))
    return gyro_prox_times, rms_prox, gyro_dist_times, rms_dist


def get_quat(data, sleeve_num=0):
    """
    Quats saved in json as Quaternion(x, y, z, w) -> used in PyQuaternion as Quaternion(w, x, y, z).
    :param data: A json file
    :param sleeve_num: int to declare which sleeve to get the data from
    :return: returns the arm and wrist quaternion values.
    """
    prox_quat = [0] * len(data['rawData']['bgxData'][sleeve_num]['Quaternion Data'][0]['data'])
    dist_quat = [0] * len(data['rawData']['bgxData'][sleeve_num]['Quaternion Data'][1]['data'])
    prox_quat_t = [0] * len(data['rawData']['bgxData'][sleeve_num]['Quaternion Data'][0]['data'])
    dist_quat_t = [0] * len(data['rawData']['bgxData'][sleeve_num]['Quaternion Data'][1]['data'])
    for i in range(0, len(data['rawData']['bgxData'][sleeve_num]['Quaternion Data'][0]['data'])):
        temp = data['rawData']['bgxData'][sleeve_num]['Quaternion Data'][0]['data'][i]['value']
        temp = [temp[0] / 2 ** 14, temp[1] / 2 ** 14, temp[2] / 2 ** 14,
                temp[3] / 2 ** 14]  # Each value is divided by 2**14
        # because the data coming from the sensor has a Q point of 14
        mag = math.sqrt(temp[0] ** 2 + temp[1] ** 2 + temp[2] ** 2 + temp[3] ** 2)
        prox_quat[i] = Quaternion(np.array([temp[3] / mag, temp[0] / mag, temp[1] / mag, temp[2] / mag]))
        prox_quat_t[i] = data['rawData']['bgxData'][sleeve_num]['Quaternion Data'][0]['data'][i][
            'sleeve local timestamp']
    for i in range(0, len(data['rawData']['bgxData'][sleeve_num]['Quaternion Data'][1]['data'])):
        temp = data['rawData']['bgxData'][sleeve_num]['Quaternion Data'][1]['data'][i]['value']
        temp = [temp[0] / 2 ** 14, temp[1] / 2 ** 14, temp[2] / 2 ** 14, temp[3] / 2 ** 14]
        mag = math.sqrt(temp[0] ** 2 + temp[1] ** 2 + temp[2] ** 2 + temp[3] ** 2)
        dist_quat[i] = Quaternion(np.array([temp[3] / mag, temp[0] / mag, temp[1] / mag, temp[2] / mag]))
        dist_quat_t[i] = data['rawData']['bgxData'][sleeve_num]['Quaternion Data'][1]['data'][i][
            'sleeve local timestamp']
    return [prox_quat_t, prox_quat, dist_quat_t, dist_quat]


def get_acc(data, sleeve_num=0):
    """
    This function is intended to extract the acceleration time and values for the
    wrist and triceps.
    :param data: A json file
    :param acc_fixed: If the acceleration is fixed in the DM and/or Firmware
    :param sleeve_num: int to declare which sleeve to get the data from
    :return: returns the acceleration times and values for the wrist and tricep.
    """
    acc_wrist_values = []
    acc_wrist_times = []
    acc_tri_values = []
    acc_tri_times = []
    for i in range(0, len(data['rawData']['bgxData'][sleeve_num]['Acceleration Data'][0]['data'])):
        acc_arr_tri = data['rawData']['bgxData'][sleeve_num]['Acceleration Data'][0]['data'][i]['value']
        acc_tri_values.append(
            [val / 256 for val in acc_arr_tri])  # Each value is divided by 2^8 because the data coming
        # from the sensor has a Q point of 8
        acc_tri_times.append(
            data['rawData']['bgxData'][sleeve_num]['Acceleration Data'][0]['data'][i]['sleeve local timestamp'])
    for i in range(0, len(data['rawData']['bgxData'][sleeve_num]['Acceleration Data'][1]['data'])):
        acc_arr_wrist = data['rawData']['bgxData'][sleeve_num]['Acceleration Data'][1]['data'][i]['value']
        acc_wrist_values.append([val / 256 for val in acc_arr_wrist])
        acc_wrist_times.append(
            data['rawData']['bgxData'][sleeve_num]['Acceleration Data'][1]['data'][i]['sleeve local timestamp'])
    return [acc_tri_times, acc_tri_values, acc_wrist_times, acc_wrist_values]


def orient_acc(data, sleeve_num=0):
    """

    :param sleeve_num:
    :param data:
    :return:
    """
    [_, prox_quat, _, dist_quat] = get_quat(data, sleeve_num=sleeve_num)
    [t_prox_acc, prox_acc, t_dist_acc, dist_acc] = get_acc(data, sleeve_num=sleeve_num)
    prox_x, dist_x, prox_y, dist_y, prox_z, dist_z, prox_t, dist_t = [], [], [], [], [], [], [], []
    shortest_len = min([len(l) for l in [prox_quat, dist_quat, prox_acc, dist_acc, t_prox_acc, t_dist_acc]])
    for i in range(shortest_len):
        temp_prox = prox_quat[i].rotate(prox_acc[i])
        temp_dist = dist_quat[i].rotate(dist_acc[i])
        prox_x.append(temp_prox[0])
        prox_y.append(temp_prox[1])
        prox_z.append(temp_prox[2])
        prox_t.append(t_prox_acc[i])
        dist_x.append(temp_dist[0])
        dist_y.append(temp_dist[1])
        dist_z.append(temp_dist[2])
        dist_t.append(t_dist_acc[i])
    return prox_t, prox_x, prox_y, prox_z, dist_t, dist_x, dist_y, dist_z


def get_most_recent_file(directory):
    files = glob.glob(directory + '/*')
    if len(files) == 0:
        return None
    most_recent_file = max(files, key=os.path.getctime)
    return most_recent_file






