#!/usr/bin/env python3

import os
import sys
import pandas as pd
import json
import csv
import cv2
import datetime
import dateutil
import glob
import scipy.io

from src.preprocessing.preprocessing_utils import *
from src.shared.cv_utils import GetVideoMetadata


def convert_timestamps(current_timestamps, first_timestamp):
    return (current_timestamps - first_timestamp)

def subject_to_fold(subject_id):
    if subject_id <= 33:
        return 'train'
    elif subject_id <= 41:
        return 'dev'
    else:
        return 'test'

def Process_UBFC(args):
    # Get the dataset metadata path.
    output_dataset_meta = os.path.join(args.out_dir, "metadata.json")

    # Loop through all sample json.
    subject_list = []
    protocol_folds = {'all': [], 'train': [], 'dev': [], 'test': []}
    for subject_name in os.listdir(args.in_dir):
        subject_id = int(subject_name.replace('subject', ''))
        input_subject_path = os.path.join(args.in_dir, subject_name)
        output_subject_id = subject_id
        output_subject_path = os.path.join(args.out_dir, str(output_subject_id))
        os.makedirs(output_subject_path, exist_ok = True)
        output_subject_meta = os.path.join(output_subject_path, "metadata.json")

        # Only one sample per subject
        # Get input paths
        sample_id = int(0)
        print("Processing subject", subject_id, "sample", sample_id)
        input_sample_path = input_subject_path
        input_phys_path = os.path.join(input_sample_path, "ground_truth.txt")
        input_video_path = os.path.join(input_sample_path, 'vid.avi')

        # Get output paths.
        output_sample_path = os.path.join(output_subject_path, str(sample_id))
        os.makedirs(output_sample_path, exist_ok = True)
        output_meta_path = os.path.join(output_sample_path, "metadata.json")
        output_phys_path = os.path.join(output_sample_path, "phys.csv")

        # Create the sample metadata.
        metadata = {}
        metadata['dataset'] = args.name
        metadata['subject'] = output_subject_id
        metadata['session'] = sample_id
        metadata['activity'] = 'sitting'

        # Read and write the phys data.
        raw_data = []
        with open(input_phys_path, "r") as phys_file:
            for line in phys_file:
                row = line.split()
                row = [float(x) for x in row]
                raw_data.append(np.array(row))
        first_timestamp = raw_data[2][0]
        phys_data = {}
        phys_data['timestamp'] = convert_timestamps(raw_data[2], first_timestamp)
        phys_data['ppg'] = raw_data[0]
        phys_data['ppg_heart_rate'] = raw_data[1]
        df = pd.DataFrame(phys_data)
        df = CorrectIrregularlySampledData(df, 30.0)
        df.to_csv(output_phys_path)

        # Determine the temporal metadata.
        phys_metadata = {}
        phys_metadata['path'] = "phys.csv"
        phys_metadata['total_entries'] = len(df['timestamp'])
        phys_metadata['sample_rate'] = 30.0
        metadata['temporal_data'] = {'phys': phys_metadata}

        # Determine the rgb metadata.
        video_metadata = GetVideoMetadata(input_video_path)
        relative_video_path = os.path.relpath(input_video_path, args.in_dir)
        video_metadata['path'] = os.path.join('data/raw/', args.name, relative_video_path)
        video_metadata['format'] = 'rgb'
        metadata['videos'] = {'main': video_metadata}

        # Determine minimum duration.
        metadata['timestamp_start'] = 0.0
        metadata['duration'] = min(phys_data['timestamp'][-1], video_metadata['temporal_data']['total_entries'] /
                                           video_metadata['temporal_data']['sample_rate'])

        # Write the metadata.json file.
        with open(output_meta_path, 'w') as meta_file:
            json.dump(metadata, meta_file)

        # Add to sample lists.
        sample_id_path = f"{output_subject_id}/{sample_id}"
        protocol_folds['all'].append(sample_id_path)
        protocol_folds[subject_to_fold(output_subject_id)].append(sample_id_path)

        # Create the subject metadata.
        metadata = {}
        metadata['dataset'] = args.name
        metadata['samples'] = [0,]

        # Write the metadata.json file.
        with open(output_subject_meta, 'w') as meta_file:
            json.dump(metadata, meta_file)
        subject_list.append(subject_id)

    # Create the dataset metadata.
    metadata = {}
    metadata['name'] = args.name
    metadata['subjects'] = subject_list
    metadata['protocols'] = {'all': protocol_folds}

    # Write the metadata.json file.
    with open(output_dataset_meta, 'w') as meta_file:
        json.dump(metadata, meta_file)
