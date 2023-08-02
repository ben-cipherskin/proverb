import gzip
import json
import glob
import os
from data_processing_folder.file_system_ops import get_all_jsons
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from pathlib2 import Path

# Specify the directory path and pattern
from visualizations import plot_fft, plot_motion
from dsp import time_to_seconds, filter_3d_signal
from grab_data import get_gyro, calculate_rms, orient_acc, get_ja

person_check = ['']
date_check = ['']
experiment_check = ["to"]  # to is analogous to sit_to_stand
save_fig = True


data_from = Path.cwd() / "data" / "raw" / "sit_to_stand"
reference_json = Path.cwd() / "references" / "devices_experiment.json"
all_exps, all_devices, exps_info = get_all_jsons(data_from, reference_json, person_check=person_check,
                                                 date_check=date_check, experiment_check=experiment_check, devices=[''])

# Iterate through each file
for i, rawData in enumerate(all_exps):
    # Grab info from all_data
    experiment = exps_info[i][0]
    date = exps_info[i][1]
    subject = exps_info[i][2]
    data = rawData[0]
    file_name = exps_info[i][-1].stem
    print("Processing file:", file_name)

    data['rawData']['sit_to_stand_count'] = 0
    location = data['rawData']['bgxData'][0]['Location']
    ja_dist_t, ja_dist, ja_prox_t, ja_prox = get_ja(data, location=location)
    prox_gyro_t, prox_gyro, dist_gyro_t, dist_gyro = get_gyro(data, sleeve_num=0)
    prox_gyro_t = [(val - prox_gyro_t[0])/1000 for val in prox_gyro_t]
    dist_gyro_t = [(val - dist_gyro_t[0])/1000 for val in dist_gyro_t]
    prox_acc_t, oriented_prox_acc, dist_acc_t, oriented_dist_acc = orient_acc(data)
    # Convert timestamp to seconds
    prox_acc_t = time_to_seconds(prox_acc_t)
    dist_acc_t = time_to_seconds(dist_acc_t)

    ## Calculate RMS Values of Gyro and Oriented Acc
    prox_gyro_rms_t, prox_gyro_rms, dist_gyro_rms_t, dist_gyro_rms = calculate_rms(prox_gyro_t, prox_gyro, dist_gyro_t, dist_gyro, filter=True)
    prox_acc_rms_t, prox_acc_rms, dist_acc_rms_t, dist_acc_rms = calculate_rms(prox_acc_t, oriented_prox_acc, dist_acc_t, oriented_dist_acc, filter=True)

    ## Plotting the FFT of the RMS Data
    # acc_fft_freq, acc_fft_vals = plot_fft(prox_acc_rms_t, prox_acc_rms, 'FFT of the Prox Acc RMS')
    # gyro_fft_freq, gyro_fft_vals = plot_fft(prox_gyro_rms_t, prox_gyro_rms, 'FFT of the Prox Gyro RMS')
    ## Plotting the FFT of the proximal joint angle data
    # TODO: Check if sampling rate is correct for joint angle data
    ja_fft_freq, acc_fft_vals = plot_fft(ja_prox_t, ja_prox, 'FFT of the Proximal Joint Angle')

    ## Filtering the motion data
    # TODO: Implement a filter function for 3d signals
    # filter_3d_signal()

    ## Plotting the motion data
    # plot_motion(prox_gyro_t, prox_gyro, dist_gyro_t, dist_gyro, "Gyro Data")
    # plot_motion(prox_acc_t, oriented_prox_acc, dist_acc_t, oriented_dist_acc, "Acc Data")

    ## Plot the RMS motion data
    # plot_motion(prox_gyro_rms_t, prox_gyro_rms, dist_gyro_rms_t, dist_gyro_rms, "RMS Gyro Data")
    # plot_motion(prox_acc_rms_t, prox_acc_rms, dist_acc_rms_t, dist_acc_rms, "RMS Acc Data")

    ## Plot the Joint Angles
    # plot_motion(ja_prox_t, ja_prox, ja_dist_t, ja_dist, "Joint Angles")

    ## Calculate the number of sit to stands
    ## Frequency analysis of proximal joint angles and RMS of proximal acc

    final_count = 0
    if 'sit_to_stand_count' in data['rawData'].keys():
        final_distance = data['rawData']['sit_to_stand_count']
        print(f'Final Count: {final_count} m')

