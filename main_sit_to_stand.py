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
from grab_data import get_gyro, get_rms, orient_acc, get_ja

directory = '/Users/cipherskin/PycharmProjects/proverb/data/manually_added/'
pattern = '*Sit_To_Stand*'

person_check = ["Suzanne1"]
date_check = ["20230710"]
experiment_check = ["sit_to_stand"]
save_fig = True


data_from = Path.cwd() / "data" / "raw" / "sit_to_stand"
reference_json = Path.cwd() / "references" / "devices_experiment.json"
all_exps, all_devices, exps_info = get_all_jsons(data_from, reference_json, person_check=person_check,
                                                 date_check=date_check, experiment_check=experiment_check, devices=[''])

# Get all files matching the pattern in the directory
files = glob.glob(os.path.join(directory, pattern))

# Iterate through each file
for file in files:
    print("Processing file:", file)
    with gzip.open(file) as fin:
        data = fin.read()
        data = json.loads(data.decode('utf-8'))

    data['rawData']['sit_to_stand_count'] = 0
    location = data['rawData']['bgxData'][0]['Location']
    ja_dist_t, ja_dist, ja_prox_t, ja_prox = get_ja(data, location=location)
    prox_gyro_t, prox_gyro, dist_gyro_t, dist_gyro = get_gyro(data, sleeve_num=0)
    prox_gyro_t = [(val - prox_gyro_t[0])/1000 for val in prox_gyro_t]
    dist_gyro_t = [(val - dist_gyro_t[0])/1000 for val in dist_gyro_t]
    acc_prox_t, oriented_prox_acc, acc_dist_t, oriented_dist_acc = orient_acc(data)
    # Convert timestamp to seconds
    acc_prox_t = time_to_seconds(acc_prox_t)
    acc_dist_t = time_to_seconds(acc_dist_t)


    # Calculate RMS Values of Gyro and Oriented Acc
    prox_gyro_rms_t, prox_gyro_rms, dist_gyro_rms_t, dist_gyro_rms = calculate_rms(prox_gyro_t, prox_gyro, dist_gyro_t, dist_gyro, filt=True)
    prox_acc_rms_t, prox_acc_rms, dist_acc_rms_t, dist_acc_rms = calculate_rms(acc_prox_t, oriented_prox_acc, oriented_dist_acc_t, oriented_dist_acc, filt=True)
    get_rms(data, gyro=False, filt=True)


    # Plotting the FFT of the RMS Data
    # plot_fft(acc_rms[0], acc_rms[1], 'FFT of the Acc RMS')
    # plot_fft(gyro_rms[0], gyro_rms[1], 'FFT of the Gyro RMS')

    # Filtering the motion data
    # filter_3d_signal()

    # Plotting the motion data
    plot_motion(prox_gyro_t, prox_gyro, dist_gyro_t, dist_gyro, "Gyro Data")
    plot_motion(acc_prox_times, oriented_prox_acc, acc_dist_times, oriented_dist_acc, "Acc Data")
    # Plot the RMS motion data
    plot_motion(prox_gyro_rms_t, prox_gyro_rms, dist_gyro_rms_t, dist_gyro_rms, "RMS Gyro Data")
    plot_motion(acc_prox_times, oriented_prox_acc, acc_dist_times, oriented_dist_acc, "RMS Acc Data")
    # Plot the Joint Angles
    plot_motion(ja_prox_t, ja_prox, ja_dist_t, ja_dist, "Joint Angles")
    break

    # fig = make_subplots(
    #     rows=7,
    #     cols=1,
    #     subplot_titles=("Proximal Acc", "Distal Acc", "Proximal Gyro", "Distal Gyro", "Gyro RMS", "Acc RMS", "Joint Angles"),
    #     shared_xaxes=True
    # )
    # acc_prox = [prox_x, prox_y, prox_z]
    # acc_dist = [dist_x, dist_y, dist_z]
    # labels = ["X", "Y", "Z"]
    # for idx, acc_data in enumerate(acc_prox):
    #     fig.add_trace(
    #         go.Scatter(
    #             x=acc_prox_times,
    #             y=acc_data,
    #             mode='lines',
    #             name=f"Prox Acc {labels[idx]}",
    #             visible=True,
    #             legendgroup=1
    #         ),
    #         row=1,
    #         col=1
    #     )
    # for idx, acc_data in enumerate(acc_dist):
    #     fig.add_trace(
    #         go.Scatter(
    #             x=acc_dist_times,
    #             y=acc_data,
    #             mode='lines',
    #             name=f"Dist Acc {labels[idx]}",
    #             visible=True,
    #             legendgroup=2
    #         ),
    #         row=2,
    #         col=1
    #     )
    # for gyro_idx in [0, 1, 2]:
    #     fig.add_trace(
    #         go.Scatter(
    #             x=prox_gyro_t,
    #             y=[val[gyro_idx] for val in prox_gyro],
    #             mode='lines',
    #             name=f"Prox Gyro {labels[gyro_idx]}",
    #             visible=True,
    #             legendgroup=3
    #         ),
    #         row=3,
    #         col=1
    #     )
    # for gyro_idx in [0, 1, 2]:
    #     fig.add_trace(
    #         go.Scatter(
    #             x=dist_gyro_t,
    #             y=[val[gyro_idx] for val in dist_gyro],
    #             mode='lines',
    #             name=f"Prox Gyro {labels[gyro_idx]}",
    #             visible=True,
    #             legendgroup=4
    #         ),
    #         row=4,
    #         col=1
    #     )
    # fig.add_trace(
    #     go.Scatter(
    #         x=gyro_rms[0],
    #         y=gyro_rms[1],
    #         mode='lines',
    #         name=f"Gyro RMS Prox",
    #         visible=True,
    #         legendgroup=5
    #     ),
    #     row=5,
    #     col=1
    # )
    # fig.add_trace(
    #     go.Scatter(
    #         x=gyro_rms[2],
    #         y=gyro_rms[3],
    #         mode='lines',
    #         name=f"Gyro RMS Dist",
    #         visible=True,
    #         legendgroup=5
    #     ),
    #     row=5,
    #     col=1
    # )
    # fig.add_trace(
    #     go.Scatter(
    #         x=acc_rms[0],
    #         y=acc_rms[1],
    #         mode='lines',
    #         name=f"Acc RMS Prox",
    #         visible=True,
    #         legendgroup=5
    #     ),
    #     row=6,
    #     col=1
    # )
    # fig.add_trace(
    #     go.Scatter(
    #         x=acc_rms[2],
    #         y=acc_rms[3],
    #         mode='lines',
    #         name=f"Acc RMS Dist",
    #         visible=True,
    #         legendgroup=5
    #     ),
    #     row=6,
    #     col=1
    # )
    # fig.add_trace(
    #     go.Scatter(
    #         x=joint_angles[0],
    #         y=joint_angles[1],
    #         mode='lines',
    #         name=f"Elbow Joint Angles",
    #         visible=True,
    #         legendgroup=5
    #     ),
    #     row=7,
    #     col=1
    # )
    #
    # fig.update(
    #     layout=dict(
    #         title=f"Sit to Stand - {file}"
    #     )
    # )
    # fig.update_layout(
    #     yaxis7=dict(range=[sum(joint_angles[1])/len(joint_angles[1])*0.9, max(joint_angles[1])*1.1])
    # )
    #
    # fig.show()

    final_count = 0
    if 'sit_to_stand_count' in data['rawData'].keys():
        final_distance = data['rawData']['sit_to_stand_count']
        print(f'Final Count: {final_count} m')

