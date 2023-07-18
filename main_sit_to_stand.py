import gzip
import json
import glob
import os

from data_processing_folder.file_system_ops import get_all_jsons
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from pathlib2 import Path

# Specify the directory path and pattern
from dsp import time_to_seconds
from grab_data import get_gyro, get_gyro_rms, orient_acc

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
    gyros = get_gyro(data, sleeve_num=0)
    prox_gyro_t, prox_gyro, dist_gyro_t, dist_gyro = gyros[0], gyros[1], gyros[2], gyros[3]
    gyro_rms = get_gyro_rms(data, filt=True)
    acc_prox_times, prox_x, prox_y, prox_z, acc_dist_times, dist_x, dist_y, dist_z = orient_acc(data)
    acc_prox_times = time_to_seconds(acc_prox_times)
    acc_dist_times = time_to_seconds(acc_dist_times)

    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=("Proximal Acc", "Distal Acc", "Proximal Gyro", "Distal Gyro"),
        shared_xaxes=True
    )
    fig.add_trace(
        go.Scatter(
            x=acc_prox_times,
            y=prox_x,
            mode='lines',
            name=f"Prox X",
            visible=True,
            legendgroup=1
        ),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=acc_prox_times,
            y=prox_y,
            mode='lines',
            name=f"Prox Y",
            visible=True,
            legendgroup=1
        ),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=acc_prox_times,
            y=prox_z,
            mode='lines',
            name=f"Prox Z",
            visible=True,
            legendgroup=1
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=acc_dist_times,
            y=dist_x,
            mode='lines',
            name=f"Dist X",
            visible=True,
            legendgroup=2
        ),
        row=2,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=acc_dist_times,
            y=dist_y,
            mode='lines',
            name=f"Dist Y",
            visible=True,
            legendgroup=2
        ),
        row=2,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=acc_dist_times,
            y=dist_z,
            mode='lines',
            name=f"Dist Z",
            visible=True,
            legendgroup=2
        ),
        row=2,
        col=1
    )

    fig.show()

    final_count = 0
    if 'sit_to_stand_count' in data['rawData'].keys():
        final_distance = data['rawData']['sit_to_stand_count']
        print(f'Final Count: {final_count} m')

