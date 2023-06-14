import json
import urllib.parse
import boto3
import gzip
import statistics

from grab_data import get_gyro, get_gyro_rms, orient_acc
from dsp import time_to_seconds
from algorithms import estimate_total_distance_gyro, split_laps
from CONSTANTS import lap_distance

#  6 MINUTE WALK TEST
#  THIS LAMBDA IS TRIGGERED WHEN A FILE IS UPLOADED TO THE s3 BUCKET digital-mirror/External_Projects/PROVERB.

print('Loading function')

s3 = boto3.client('s3')

bucket = 'digital-mirror'


def lambda_handler(event, context):
    # print("Received event: " + json.dumps(event, indent=2))

    # Get the filename
    specified_key = event['Records'][0]['s3']['object']['key']
    temp_key_time = specified_key.split("T")[-1]
    temp_key_time_len = len(temp_key_time)
    temp_key_time = temp_key_time.replace("_", ":")
    specified_key = specified_key[:-temp_key_time_len] + temp_key_time
    key = urllib.parse.unquote_plus(specified_key, encoding='utf-8')

    print(f'Key: {key}')

    if '6_Minute_Walk_Test' in str(key):
        pass
    else:
        return {
            'statusCode': 200,
            'body': 'Not a 6MWT Recording'
        }

    try:
        # Fetch the file from s3
        response = s3.get_object(Bucket=bucket, Key=key)

        # Deserialize the file's content
        temp = response["Body"]
        with gzip.open(temp) as fin:
            data = fin.read()
            data = json.loads(data.decode('utf-8'))

        if '6MWT_Distance' in data['rawData'].keys():
            final_distance = data['rawData']['6MWT_Distance']
            print(f'Distance: {final_distance} m')
            return {
                'statusCode': 200,
                'body': f'Distance Succesfully uploaded to recording file.'
            }

        data['rawData']['6MWT_Distance'] = 0
        gyros = get_gyro(data, sleeve_num=0)
        prox_gyro_t, prox_gyro, dist_gyro_t, dist_gyro = gyros[0], gyros[1], gyros[2], gyros[3]
        gyro_rms = get_gyro_rms(data, filt=True)
        acc_prox_times, prox_x, prox_y, prox_z, acc_dist_times, dist_x, dist_y, dist_z = orient_acc(data)
        acc_prox_times = time_to_seconds(acc_prox_times)
        acc_dist_times = time_to_seconds(acc_dist_times)

        peak_distance_prox, peak_distance_dist, sorted_turns_prox, sorted_turns_dist = estimate_total_distance_gyro(
            gyro_rms, gyros)

        steps_in_each_turn, avg_steps, final_lap_steps, freq_in_each_turn, time_in_each_turn = split_laps(
            acc_dist_times, dist_x, dist_y, dist_z, gyro_rms[2], sorted_turns_dist)

        steps_by_freq_in_each_turn = [int(a * b) for a, b in zip(time_in_each_turn,
                                                                 freq_in_each_turn)]  # Multiply the max frequency in each turn by the time for each turn
        avg_distance_per_step_per_turn_by_freq = lap_distance / statistics.mean(steps_by_freq_in_each_turn[
                                                                                :-1])  # Lap distance divided by the average number of steps for each lap (not including last lap)
        total_dist_freq_est = (lap_distance * len(sorted_turns_dist)) + (
                    time_in_each_turn[-1] * freq_in_each_turn[-1] * avg_distance_per_step_per_turn_by_freq)
        total_dist_time_ratio = lap_distance * (
                    len(sorted_turns_dist) + (time_in_each_turn[-1] / statistics.mean(time_in_each_turn[:-1])))

        # Print the content
        print(f"Estimated Distance (Time Ratio): {total_dist_time_ratio}")
        print(f"Estimated Distance (Freq Est): {total_dist_freq_est}")
        averaged_distance = round((total_dist_freq_est + total_dist_time_ratio) / 2)

        data['rawData']['6MWT_Distance'] = averaged_distance
        data_to_zip = json.dumps(data).encode('utf-8')
        zipped_data = gzip.compress(data_to_zip)

        s3.put_object(Bucket=bucket, Key=key, Body=zipped_data)

        return {
            'statusCode': 200,
            'body': '6MWT Distance Calculated Succesfully and file re-uploaded.'
        }


    except Exception as e:
        # data['rawData']['6MWT_Distance'] = 0
        # data_to_zip = json.dumps(data).encode('utf-8')
        # zipped_data = gzip.compress(data_to_zip)
        # s3.put_object(Bucket=bucket, Key=key, Body=zipped_data)
        print(e)
        raise e



