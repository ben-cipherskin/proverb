"""
Script used to pull the data from AWS and to be put in in this repository /data/... Both data from the sleeves and
peripheral devices will be pulled.  This is data is needed for the main programs to work. It pulls the data directly
from the AWS, no local syncing.

It also syncs the chronojumps database to AWS and saves it

Last updated: 2022-04-01
by: Simon Grannetia
"""
from pathlib2 import Path
import boto3

from data_processing_folder.file_system_ops import make_folder
from data_processing_folder.parsing_zip import data_pull

s3 = boto3.resource('s3')

# Variables
check_if_exists = True
pull_experiment = ['6MWT']
all_data_to = Path.cwd() / 'data' / 'raw'
make_folder(all_data_to)
max_exp_to_grab = 20
periph_bucket_name = 'data-science-peripheral-devices'
periph_data = s3.Bucket(periph_bucket_name)  # Amazon s3 bucket with all data for peripheral devices
sleeve_data = s3.Bucket('digital-mirror')  # S3 bucket with sleeve data
all_data_to = Path.cwd() / 'data' / 'raw'   # where all the files are stored in file structure locally
reference_file = Path.cwd() / 'references' / 'files_to_grab.json'
data_pull(pull_experiment, max_exp_to_grab, check_if_exists, all_data_to=all_data_to, reference_file=reference_file,
          experiment_types=pull_experiment, devices=['sleeve'])

# Pushing peripheral data if not on aws
# upload_to_aws(periph_data, all_data_to, [''])
