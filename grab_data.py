import numpy as np
import math
import os
import glob
import scipy.interpolate as interp
from scipy.signal import detrend
from scipy import signal
from pyquaternion import Quaternion
from dsp import time_to_seconds, interpolate_to_hz, butter_lowpass, transform_global_t


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


def calculate_rms(prox_t, prox, dist_t, dist, filter=False):
    """
    function to calculate rms of gyro data and optionally filter it

    :param prox_t: list; timestamps of proximal data
    :param prox: list; proximal data
    :param dist_t: list; timestamps of distal data
    :param dist: list; distal data
    :param filter: bool; whether or not to filter data
    :return: prox_t, prox_rms, dist_t, dist_rms: lists; calculated rms data and altered timestamps (if filtered)
    """
    # Extracting signal components
    prox_x = [val[0] for val in prox]
    prox_y = [val[1] for val in prox]
    prox_z = [val[2] for val in prox]

    dist_x = [val[0] for val in dist]
    dist_y = [val[1] for val in dist]
    dist_z = [val[2] for val in dist]

    if filter:
        highcut = 0.15  # Hz
        fs = 100  # Hz
        order = 2
        # Error handling for cases where timestamps have multiple zero starts
        if prox_t[1] == 0.0:
            prox_t[1] = 0.01
        _, prox_x = interpolate_to_hz(prox_t, prox_x, fs, detrend_signal=True, order=1)
        _, prox_y = interpolate_to_hz(prox_t, prox_y, fs, detrend_signal=True, order=1)
        prox_t, prox_z = interpolate_to_hz(prox_t, prox_z, fs, detrend_signal=True, order=1)
        # Error handling for cases where timestamps have multiple zero starts
        if dist_t[1] == 0.0:
            dist_t[1] = 0.01
        _, dist_x = interpolate_to_hz(dist_t, dist_x, fs, detrend_signal=True, order=1)
        _, dist_y = interpolate_to_hz(dist_t, dist_z, fs, detrend_signal=True, order=1)
        dist_t, dist_z = interpolate_to_hz(dist_t, dist_z, fs, detrend_signal=True, order=1)

        # Applying lowpass filter to quaternion components
        prox_x = (butter_lowpass(prox_x, highcut=highcut, fs=fs, order=order))
        prox_y = (butter_lowpass(prox_y, highcut=highcut, fs=fs, order=order))
        prox_z = (butter_lowpass(prox_z, highcut=highcut, fs=fs, order=order))

        dist_x = (butter_lowpass(dist_x, highcut=highcut, fs=fs, order=order))
        dist_y = (butter_lowpass(dist_y, highcut=highcut, fs=fs, order=order))
        dist_z = (butter_lowpass(dist_z, highcut=highcut, fs=fs, order=order))

    # Initialize empty mutable arrays for RMS values
    rms_prox = [0] * len(prox_x)
    rms_dist = [0] * len(dist_x)
    for j in range(len(prox_x)):
        rms_prox[j] = math.sqrt(((prox_x[j] ** 2) + (prox_y[j] ** 2) +
                                   (prox_z[j] ** 2)) / 3)
    for j in range(len(dist_x)):
        rms_dist[j] = math.sqrt(((dist_x[j] ** 2) + (dist_y[j] ** 2) +
                                   (dist_z[j] ** 2)) / 3)
    return prox_t, rms_prox, dist_t, rms_dist



def get_rms(data, gyro=True, filt=False):
    """

    :param filt:
    :param data:
    :return:
    """
    if gyro:
        grabbed_data = get_gyro(data, sleeve_num=0)
    else:
        grabbed_data = get_acc(data, sleeve_num=0)
    prox_t, prox_data, dist_t, dist_data = grabbed_data[0], grabbed_data[1], grabbed_data[2], grabbed_data[3]

    # Extracting quaternion components
    prox_x = [val[0] for val in prox_data]
    prox_y = [val[1] for val in prox_data]
    prox_z = [val[2] for val in prox_data]

    dist_x = [val[0] for val in dist_data]
    dist_y = [val[1] for val in dist_data]
    dist_z = [val[2] for val in dist_data]

    # Putting data in dictionary for plotting purposes
    prox_times = time_to_seconds(prox_t)
    dist_times = time_to_seconds(dist_t)

    if filt:
        highcut = 0.15  # Hz
        fs = 100  # Hz
        order = 2
        # Error handling for cases where timestamps have multiple zero starts
        if prox_times[1] == 0.0:
            prox_times[1] = 0.01
        prox_xt, prox_x = interpolate_to_hz(prox_times, prox_x, fs, detrend_signal=True, order=1)
        prox_yt, prox_y = interpolate_to_hz(prox_times, prox_y, fs, detrend_signal=True, order=1)
        prox_times, prox_z = interpolate_to_hz(prox_times, prox_z, fs, detrend_signal=True, order=1)
        # Error handling for cases where timestamps have multiple zero starts
        if dist_times[1] == 0.0:
            dist_times[1] = 0.01
        dist_xt, dist_x = interpolate_to_hz(dist_times, dist_x, fs, detrend_signal=True, order=1)
        dist_yt, dist_y = interpolate_to_hz(dist_times, dist_y, fs, detrend_signal=True, order=1)
        dist_times, dist_z = interpolate_to_hz(dist_times, dist_z, fs, detrend_signal=True, order=1)

        # Applying lowpass filter to quaternion components
        prox_x = (butter_lowpass(prox_x, highcut=highcut, fs=fs, order=order))
        prox_y = (butter_lowpass(prox_y, highcut=highcut, fs=fs, order=order))
        prox_z = (butter_lowpass(prox_z, highcut=highcut, fs=fs, order=order))

        dist_x = (butter_lowpass(dist_x, highcut=highcut, fs=fs, order=order))
        dist_y = (butter_lowpass(dist_y, highcut=highcut, fs=fs, order=order))
        dist_z = (butter_lowpass(dist_z, highcut=highcut, fs=fs, order=order))

    rms_prox = []
    rms_dist = []
    for j in range(len(prox_x)):
        rms_prox.append(math.sqrt(((prox_x[j] ** 2) + (prox_y[j] ** 2) +
                                   (prox_z[j] ** 2)) / 3))
    for j in range(len(dist_x)):
        rms_dist.append(math.sqrt(((dist_x[j] ** 2) + (dist_y[j] ** 2) +
                                   (dist_z[j] ** 2)) / 3))
    return prox_times, rms_prox, dist_times, rms_dist


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

    shortest_len = min([len(l) for l in [prox_quat, dist_quat, prox_acc, dist_acc, t_prox_acc, t_dist_acc]])
    # Declare empty mutable oriented arrays of length shortest_len
    oriented_prox = [0] * shortest_len
    oriented_dist = [0] * shortest_len
    oriented_prox_t = [0] * shortest_len
    oriented_dist_t = [0] * shortest_len
    for i in range(shortest_len):
        oriented_prox_t[i] = t_prox_acc[i]
        oriented_prox[i] = prox_quat[i].rotate(prox_acc[i])
        oriented_dist_t[i] = t_dist_acc[i]
        oriented_dist[i] = dist_quat[i].rotate(dist_acc[i])
    return oriented_prox_t, oriented_prox, oriented_dist_t, oriented_dist


def get_ja(data, location, sleeve_num=0):
    """

    Parameters
    ----------
    data: dict; containing all the sleeve data
    location: str; specifying the location of the joint getting the data for.

    Returns 4 array containing: time stamps for joint angles, joint angles, time stamps for joint angle rates, joint angle rates
    -------

    """
    t_ja, ja, t_ja_prox, ja_prox = [], [], [], []
    for file_location_angles_dict in data['rawData']['metrics']['Recorded Angles']:
        if type(file_location_angles_dict['Joint Location Name']) == int:
            if location == file_location_angles_dict['Joint Location Name']:
                pass
            else:
                continue
        try:
            if location in file_location_angles_dict["Joint Location Name"]:
                for angles in file_location_angles_dict['Joint Angles']:
                    t_ja.append(angles['time'])
                    ja.append(angles['value'])
        except TypeError:  # This try and except function is needed for a bug in 0.755.4 DM release
            if type(file_location_angles_dict["Joint Location Name"]) == int:
                for angles in file_location_angles_dict['Joint Angles']:
                    t_ja.append(angles['time'])
                    ja.append(angles['value'])

        try:
            if location in file_location_angles_dict["Joint Location Name"]:
                for angles in file_location_angles_dict['Proximal Limb Flexion Extension Angles']:
                    t_ja_prox.append(angles['time'])
                    ja_prox.append(angles['value'])
        except TypeError:  # This try and except function is needed for a bug in 0.755.4 DM release
            if type(file_location_angles_dict["Joint Location Name"]) == int:
                for angles in file_location_angles_dict['Proximal Limb Flexion Extension Angles']:
                    t_ja_prox.append(angles['time'])
                    ja_prox.append(angles['value'])
    return transform_global_t(t_ja), ja, transform_global_t(t_ja_prox), ja_prox


def get_prox_ja_w_time(data, location, sleeve_num=0):
    """

    Parameters
    ----------
    data: dict; containing all the sleeve data
    location: str; specifying the location of the joint getting the data for.

    Returns 4 array containing: time stamps for proximal joint angles, proximal joint angles, time stamps for proximal joint angle rates, proximal joint angle rates
    -------

    """
    # TODO: Remove this when a new app is pushed that resolves this issue
    if type(data['rawData']['metrics']['Recorded Angles'][0]['Joint Location Name']) == int:
        if 'Right' in location:
            location = 2
        elif 'Left' in location:
            location = 4
    t_ja, ja, t_jar, jar = [], [], [], []
    for file_location_angles_dict in data['rawData']['metrics']['Recorded Angles']:
        if type(file_location_angles_dict['Joint Location Name']) == int:
            if file_location_angles_dict['Joint Location Name'] == location:
                pass
            else:
                continue
        try:
            if location in file_location_angles_dict["Joint Location Name"]:
                for angles in file_location_angles_dict['Proximal Limb Flexion Extension Angles']:
                    t_ja.append(angles['time'])
                    ja.append(angles['value'])
                for joint_angle_rates in file_location_angles_dict['Proximal Limb Flexion Extension Angle Rates']:
                    t_jar.append(joint_angle_rates['time'])
                    jar.append(joint_angle_rates['value'])
        except TypeError:  # This try and except function is needed for a bug in 0.755.4 DM release
            if type(file_location_angles_dict["Joint Location Name"]) == int:
                for angles in file_location_angles_dict['Proximal Limb Flexion Extension Angles']:
                    t_ja.append(angles['time'])
                    ja.append(angles['value'])
                for joint_angle_rates in file_location_angles_dict['Proximal Limb Flexion Extension Angle Rates']:
                    t_jar.append(joint_angle_rates['time'])
                    jar.append(joint_angle_rates['value'])
    return transform_global_t(t_ja), ja, transform_global_t(t_jar), jar


def get_most_recent_file(directory):
    files = glob.glob(directory + '/*')
    if len(files) == 0:
        return None
    most_recent_file = max(files, key=os.path.getctime)
    return most_recent_file






