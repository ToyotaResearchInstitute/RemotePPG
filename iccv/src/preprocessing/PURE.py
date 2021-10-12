#!/usr/bin/env python3

import os
import sys
import pandas as pd
import json
import csv
from src.preprocessing.preprocessing_utils import *
import cv2
import datetime
import dateutil
from zipfile38 import ZipFile
import glob


def convert_timestamp(current_timestamp, first_timestamp):
    return float(current_timestamp - first_timestamp) * 1e-9

def Process_PURE(args):
    # Setup dataset constants
    activity_map = {}
    activity_map[1] = 'sitting'
    activity_map[2] = 'talking'
    activity_map[3] = 'slow_translation'
    activity_map[4] = 'fast_translation'
    activity_map[5] = 'small_rotation'
    activity_map[6] = 'medium_rotation'
    activity_map[7] = 'talking'

    # Get the dataset metadata path.
    output_dataset_meta = os.path.join(args.out_dir, "metadata.json")

    # Unzip all the data.
    for zip_path in glob.glob(os.path.join(args.in_dir, "*.zip")):
        with ZipFile(zip_path, 'r') as zip_file:
            zip_file.extractall(args.in_dir)
        os.remove(zip_path)

    # Loop through all sample json.
    subject_list = list(range(1,11))
    for subject_id in subject_list:
        output_subject_path = os.path.join(args.out_dir, str(subject_id))
        output_subject_meta = os.path.join(output_subject_path, "metadata.json")
        os.makedirs(output_subject_path, exist_ok = True)
        if subject_id == 6:
            sample_list = [1, 3, 4, 5, 6]  # Subject six is missing the second session.
        elif subject_id == 7:
            sample_list = list(range(1,8))
        else:
            sample_list = list(range(1,7))
        for sample_id in sample_list:
            print("Processing subject", subject_id, "sample", sample_id)
            
            sample_basename = '{:02d}-{:02d}'.format(subject_id, sample_id)
            input_video_path = os.path.join(args.in_dir, sample_basename)
            output_sample_path = os.path.join(output_subject_path, str(sample_id))
            os.makedirs(output_sample_path, exist_ok = True)

            # Read the metadata.
            input_meta_path = input_video_path + ".json"
            output_meta_path = os.path.join(output_sample_path, "metadata.json")
            output_main_path = os.path.join(output_sample_path, "main.csv")
            output_phys_path = os.path.join(output_sample_path, "phys.csv")
            with open(input_meta_path, 'r') as input_meta_file:
                json_data = json.load(input_meta_file)

                # Determine starting timestamp for entire sample.
                ids = ['/Image', '/FullPackage']
                first_timestamp = min([min([entry['Timestamp'] for entry in json_data[id]]) for id in ids])
                
                # Write the main.csv file.
                frame_data = {}
                frame_data['timestamp'] = [convert_timestamp(x['Timestamp'], first_timestamp) for x in json_data['/Image']]
                df = pd.DataFrame(frame_data)
                df.to_csv(output_main_path)

                # Write the phys.csv file.
                phys_data = {'timestamp': [], 'ppg': [], 'ppg_heart_rate': [], 'o2sat': [], 'signal_quality': []}
                for entry in json_data['/FullPackage']:
                    phys_data['timestamp'].append(convert_timestamp(entry['Timestamp'], first_timestamp))
                    phys_data['ppg'].append(entry['Value']['waveform'])
                    phys_data['ppg_heart_rate'].append(entry['Value']['pulseRate'])
                    phys_data['o2sat'].append(entry['Value']['o2saturation'])
                    phys_data['signal_quality'].append(entry['Value']['signalStrength'] / 5.0)
                df = pd.DataFrame(phys_data)
                df = CorrectIrregularlySampledData(df, 30.0)
                df.to_csv(output_phys_path)
                
                # Create the sample metadata.
                metadata = {}
                metadata['dataset'] = args.name
                metadata['subject'] = subject_id
                metadata['lighting'] = 'natural'
                metadata['session'] = sample_id
                metadata['activity'] = activity_map[sample_id]

                # Determine the video metadata.
                video_metadata = {}
                relative_video_path = os.path.relpath(input_video_path, args.in_dir)
                video_metadata['path'] = os.path.join('data/raw/', args.name, relative_video_path)
                video_metadata['format'] = 'rgb'
                video_metadata['width'] = 640
                video_metadata['height'] = 480
                video_metadata['temporal_data'] = {}
                video_metadata['temporal_data']['path'] = "main.csv"
                video_metadata['temporal_data']['total_entries'] = len(frame_data['timestamp'])
                video_metadata['temporal_data']['sample_rate'] = video_metadata['temporal_data']['total_entries'] / (frame_data['timestamp'][-1] - frame_data['timestamp'][0])
                metadata['videos'] = {'main': video_metadata}

                # Determine the temporal metadata.
                phys_metadata = {}
                phys_metadata['path'] = "phys.csv"
                phys_metadata['total_entries'] = len(df['timestamp'])
                phys_metadata['sample_rate'] = 30.0
                metadata['temporal_data'] = {'phys': phys_metadata}

                # Determine minimum duration.
                metadata['timestamp_start'] = max(phys_data['timestamp'][0], frame_data['timestamp'][0])
                metadata['duration'] = min(phys_data['timestamp'][-1], frame_data['timestamp'][-1]) - metadata['timestamp_start']
                                       
                # Write the metadata.json file.
                with open(output_meta_path, 'w') as meta_file:
                    json.dump(metadata, meta_file)

        # Create the subject metadata.
        metadata = {}
        metadata['dataset'] = args.name
        metadata['samples'] = sample_list

        # Write the metadata.json file.
        with open(output_subject_meta, 'w') as meta_file:
            json.dump(metadata, meta_file)

    # Create the dataset metadata.
    metadata = {}
    metadata['name'] = args.name
    metadata['subjects'] = subject_list

    # Read the protocol files (if exist).
    if args.protocols:
        metadata['protocols'] = LoadProtocols(args.protocols)

    # Write the metadata.json file.
    with open(output_dataset_meta, 'w') as meta_file:
        json.dump(metadata, meta_file)
