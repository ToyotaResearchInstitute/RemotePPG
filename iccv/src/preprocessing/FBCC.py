#!/usr/bin/env python3

import os
import sys
import pandas as pd
import json
import csv
import datetime
import dateutil
import boto3
import tempfile

from src.shared.s3_utils import split_s3_path
from src.shared.cv_utils import GetVideoMetadata


def Process_FBCC(args):
    # List of samples with rate issues
    doubled_samples = ['CasualConversationsNO/0208/0208_00.MP4', 
                       'CasualConversationsNO/0201/0201_00.MP4']

    # Open connection to S3
    s3_client = boto3.client('s3')

    # Get the dataset metadata path.
    output_dataset_meta = os.path.join(args.out_dir, "metadata.json")

    # Get the dataset metadata path.
    input_dataset_meta = os.path.join(args.in_dir, "CasualConversations.json")

    # Read the dataset annotation json
    with open(input_dataset_meta, 'r') as input_meta_file:
        json_data = json.load(input_meta_file)

    # Loop through subjects
    subject_list = []
    for subject_name, subject_json in json_data.items():
        subject_id = int(subject_name)
        subject_list.append(subject_id)
        output_subject_path = os.path.join(args.out_dir, str(subject_id))
        output_subject_meta = os.path.join(output_subject_path, "metadata.json")
        os.makedirs(output_subject_path, exist_ok = True)

        # Loop through samples
        sample_list = []
        for sample_id, sample_path in enumerate(subject_json['files']):
            print("Processing subject", subject_id, "sample", sample_id)
            sample_list.append(sample_id)
            output_sample_path = os.path.join(output_subject_path, str(sample_id))
            os.makedirs(output_sample_path, exist_ok = True)
            output_meta_path = os.path.join(output_sample_path, "metadata.json")

            # Create the sample metadata.
            metadata = {}
            metadata['dataset'] = args.name
            metadata['subject'] = subject_id
            metadata['session'] = sample_id
            metadata['lighting'] = 'dark' if sample_path in subject_json['dark_files'] else 'natural'

            # Determine the video metadata.
            video_path = os.path.join('s3://tri-guardian-drivernet/projects/P03-RPPG/data/FBCC/', sample_path)
            s3_bucket, s3_key = split_s3_path(video_path)
            try:
                s3_resp = s3_client.get_object(Bucket=s3_bucket, Key=s3_key, Range='bytes=0-1000000')
                s3_data = s3_resp['Body'].read()
                with tempfile.NamedTemporaryFile() as fp:
                    fp.write(s3_data)
                    video_metadata = GetVideoMetadata(fp.name)
                video_metadata['format'] = 'rgb'
                video_metadata['path'] = video_path
                # Check for videos with metadata doubling issue
                if sample_path in doubled_samples:
                    for key in ['total_entries', 'sample_rate']:
                        video_metadata[key] = video_metadata[key] / 2
                metadata['videos'] = {'main': video_metadata}
            except:
                metadata['videos'] = {}

            # Write the metadata.json file.
            with open(output_meta_path, 'w') as meta_file:
                json.dump(metadata, meta_file)

        # Gather the subject demographic data
        subject_label = subject_json["label"]
        subject_demographics = {}
        try:
            subject_demographics['age'] = int(subject_label['age'])
        except:
            subject_demographics['age'] = None
        subject_demographics['gender'] = subject_label['gender'].lower()
        subject_demographics['skin_tone'] = int(subject_label['skin-type'])

        # Create the subject metadata.
        metadata = {}
        metadata['dataset'] = args.name
        metadata['samples'] = sample_list
        metadata['demographics'] = subject_demographics

        # Write the metadata.json file.
        with open(output_subject_meta, 'w') as meta_file:
            json.dump(metadata, meta_file)

    # Create the dataset metadata.
    metadata = {}
    metadata['name'] = args.name
    metadata['subjects'] = subject_list

    # Write the metadata.json file.
    with open(output_dataset_meta, 'w') as meta_file:
        json.dump(metadata, meta_file)
