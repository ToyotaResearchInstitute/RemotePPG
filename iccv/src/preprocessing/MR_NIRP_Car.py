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
import scipy.io


def convert_timestamps(current_timestamps, first_timestamp):
    return (current_timestamps - first_timestamp)

def subject_to_fold(subject_id):
    if subject_id <= 12:
        return 'train'
    elif subject_id <= 15:
        return 'dev'
    else:
        return 'test'

def Process_MR_NIRP_Car(args):
    # Setup dataset constants
    activity_map = {"garage": {"still": "sitting", "small": "small_motion"}, 
                "driving": {"still": "driving", "small": "driving_small_motion"}}

    # Get the dataset metadata path.
    output_dataset_meta = os.path.join(args.out_dir, "metadata.json")

    # Unzip all the data.
    for zip_path in glob.glob(os.path.join(args.in_dir, "Subject*/*/*.zip")):
        base_path = os.path.dirname(zip_path)
        with ZipFile(zip_path, 'r') as zip_file:
            zip_file.extractall(base_path)
        os.remove(zip_path)

    # Loop through all sample json.
    protocol_folds = {'all': [], 'train': [], 'dev': [], 'test': []}
    subject_list = list(range(1,20))
    for subject_id in subject_list:
        input_subject_path = os.path.join(args.in_dir, 'Subject{:d}'.format(subject_id))
        output_subject_id = subject_id
        if subject_id == 16:
            output_subject_id = 2 # Subject 16 is actually also subject 2, but at night
        output_subject_path = os.path.join(args.out_dir, str(output_subject_id))
        if subject_id != 16:
            os.makedirs(output_subject_path, exist_ok = True)
        output_subject_meta = os.path.join(output_subject_path, "metadata.json")

        # Get the sample paths.
        for sample_id, sample_name in enumerate(os.listdir(input_subject_path)):
            # Skip subject2_garage_small_motion_940 - RGB corrupted
            if 'subject2_garage_small_motion_940' in sample_name:
                continue
            print("Processing subject", subject_id, "sample", sample_id)

            # Correct for subject 16 (subject 2)
            if subject_id == 16:
                sample_id += 4

            # Get input paths
            input_sample_path = os.path.join(input_subject_path, sample_name)
            sample_parts = sample_name.split("_")
            input_phys_path = os.path.join(input_sample_path, "PulseOX/pulseOx.mat")
            input_rgb_video_path = os.path.join(input_sample_path, 'RGB')
            input_nir_video_path = os.path.join(input_sample_path, 'NIR')

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
            metadata['location'] = sample_parts[1]
            metadata['activity'] = activity_map[sample_parts[1]][sample_parts[2]]
            metadata['nir_wavelength'] = int(sample_parts[-1])

            # Read and write the phys data.
            raw_data = scipy.io.loadmat(input_phys_path)
            first_timestamp = raw_data['pulseOxTime'][0][0]
            phys_data = {}
            phys_data['timestamp'] = convert_timestamps(raw_data['pulseOxTime'][0], first_timestamp)
            phys_data['ppg'] = raw_data['pulseOxRecord'][0]
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
            num_frames = len(os.listdir(input_rgb_video_path))
            rgb_metadata = {}
            relative_video_path = os.path.relpath(input_rgb_video_path, args.in_dir)
            rgb_metadata['path'] = os.path.join('data/raw/', args.name, relative_video_path)
            rgb_metadata['format'] = 'bayer'
            rgb_metadata['width'] = 640
            rgb_metadata['height'] = 640
            rgb_metadata['temporal_data'] = {}
            rgb_metadata['temporal_data']['total_entries'] = num_frames
            rgb_metadata['temporal_data']['sample_rate'] = 30.0
            
            # Determine the nir metadata.
            num_frames = len(os.listdir(input_nir_video_path))
            nir_metadata = {}
            relative_video_path = os.path.relpath(input_nir_video_path, args.in_dir)
            nir_metadata['path'] = os.path.join('data/raw/', args.name, relative_video_path)
            nir_metadata['format'] = 'nir_alternating'
            nir_metadata['width'] = 640
            nir_metadata['height'] = 640
            nir_metadata['temporal_data'] = {}
            nir_metadata['temporal_data']['total_entries'] = int(np.floor(num_frames / 2))
            nir_metadata['temporal_data']['sample_rate'] = 30.0
            
            metadata['videos'] = {'main': rgb_metadata, 'nir': nir_metadata}

            # Determine minimum duration.
            metadata['timestamp_start'] = 0.0
            rgb_duration = rgb_metadata['temporal_data']['total_entries'] / rgb_metadata['temporal_data']['sample_rate']
            nir_duration = nir_metadata['temporal_data']['total_entries'] / nir_metadata['temporal_data']['sample_rate']
            metadata['duration'] = min(phys_data['timestamp'][-1], rgb_duration, nir_duration)

            # Write the metadata.json file.
            with open(output_meta_path, 'w') as meta_file:
                json.dump(metadata, meta_file)

            # Add to sample lists.
            sample_id_path = f"{output_subject_id}/{sample_id}"
            protocol_folds['all'].append(sample_id_path)
            protocol_folds[subject_to_fold(output_subject_id)].append(sample_id_path)

        # Create the subject metadata.
        if subject_id != 16:
            metadata = {}
            metadata['dataset'] = args.name
            metadata['samples'] = list(range(4)) if subject_id != 2 else [0, 2, 3, 4, 5]

            # Write the metadata.json file.
            with open(output_subject_meta, 'w') as meta_file:
                json.dump(metadata, meta_file)

    # Create the dataset metadata.
    metadata = {}
    metadata['name'] = args.name
    metadata['subjects'] = list(range(1,16)) + list(range(17,20))
    metadata['protocols'] = {'all': protocol_folds}

    # Write the metadata.json file.
    with open(output_dataset_meta, 'w') as meta_file:
        json.dump(metadata, meta_file)
