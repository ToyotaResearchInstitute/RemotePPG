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


def subject_to_fold(subject_id):
    if subject_id <= 34:
        return 'train'
    elif subject_id <= 45:
        return 'dev'
    else:
        return 'test'

def Process_UBFC_Phys(args):
    # Setup dataset constants
    activity_map = {1: "sitting", 
                    2: "talking",
                    3: "sitting_math"}

    # Get the dataset metadata path.
    output_dataset_meta = os.path.join(args.out_dir, "metadata.json")

    # Unzip all the data.
    for zip_path in glob.glob(os.path.join(args.in_dir, "*.zip")):
        with ZipFile(zip_path, 'r') as zip_file:
            zip_file.extractall(args.in_dir)
        os.remove(zip_path)

    # Loop through all sample json.
    subject_list = list(range(1, 57))
    protocol_folds = {'all': [], 'train': [], 'dev': [], 'test': []}
    for subject_id in subject_list:
        input_subject_path = os.path.join(args.in_dir, 's' + str(subject_id))
        output_subject_path = os.path.join(args.out_dir, str(subject_id))
        os.makedirs(output_subject_path, exist_ok = True)
        output_subject_meta = os.path.join(output_subject_path, "metadata.json")

        # Loop through samples (each type of test)
        sample_list = [1, 2, 3]
        for sample_id in sample_list:
            # Get input paths
            print("Processing subject", subject_id, "sample", sample_id)
            sample_basename = 's{:d}_T{:d}'.format(subject_id, sample_id)
            input_bvp_path = os.path.join(input_subject_path, "bvp_" + sample_basename + '.csv')
            input_eda_path = os.path.join(input_subject_path, "eda_" + sample_basename + '.csv')
            input_video_path = os.path.join(input_subject_path, "vid_" + sample_basename + '.avi')

            # Get output paths.
            output_sample_path = os.path.join(output_subject_path, str(sample_id))
            os.makedirs(output_sample_path, exist_ok = True)
            output_meta_path = os.path.join(output_sample_path, "metadata.json")
            output_phys_path = os.path.join(output_sample_path, "phys.csv")

            # Create the sample metadata.
            metadata = {}
            metadata['dataset'] = args.name
            metadata['subject'] = subject_id
            metadata['session'] = sample_id
            metadata['activity'] = activity_map[sample_id]

             # Read and write the phys data.
            phys_data = {}
            with open(input_bvp_path, 'r') as bvp_file:
                bvp_data = bvp_file.readlines()
                bvp_data = [float(x.strip()) for x in bvp_data]
                phys_data['bvp'] = bvp_data  #  64 Hz
            with open(input_eda_path, 'r') as eda_file:
                eda_data = eda_file.readlines()
                eda_data = [float(x.strip()) for x in eda_data]
                eda_data = list(np.repeat(eda_data, 16))  # Convert to 64 Hz
                phys_data['eda'] = eda_data
            phys_data['timestamp'] = np.arange(len(phys_data['bvp'])) / 64
            df = pd.DataFrame(phys_data)
            df = CorrectIrregularlySampledData(df, 30.0)
            df.to_csv(output_phys_path)

            # Determine the temporal metadata.
            phys_metadata = {}
            phys_metadata['path'] = "phys.csv"
            phys_metadata['total_entries'] = len(df['timestamp'])
            phys_metadata['sample_rate'] = 30.0
            metadata['temporal_data'] = {'phys': phys_metadata}

            # Determine the video metadata.
            video_metadata = GetVideoMetadata(input_video_path)
            relative_video_path = os.path.relpath(input_video_path, args.in_dir)
            video_metadata['path'] = os.path.join('data/raw/', args.name, relative_video_path)
            video_metadata['format'] = 'rgb'
            video_metadata['temporal_data']['total_entries'] -= 1  # Drop last frame, due to corruption
            metadata['videos'] = {'main': video_metadata}

            # Determine minimum duration.
            metadata['timestamp_start'] = 0.0
            metadata['duration'] = min(phys_data['timestamp'][-1], video_metadata['temporal_data']['total_entries'] /
                                            video_metadata['temporal_data']['sample_rate'])

            # Write the metadata.json file.
            with open(output_meta_path, 'w') as meta_file:
                json.dump(metadata, meta_file)

        # Create the subject metadata.
        metadata = {}
        metadata['dataset'] = args.name
        metadata['samples'] = sample_list

        # Read the info metadata.
        input_info_path = os.path.join(input_subject_path, 'info_s{:d}.txt'.format(subject_id))
        with open(input_info_path, 'r') as info_file:
            info_file.readline()
            info_sex = info_file.readline().strip()  #  m or f
            metadata['demographics'] = {'gender': 'male' if info_sex == 'm' else 'female'}

        # Write the metadata.json file.
        with open(output_subject_meta, 'w') as meta_file:
            json.dump(metadata, meta_file)

        # Add to sample lists.
        sample_id_path = f"{subject_id}/{sample_id}"
        protocol_folds['all'].append(sample_id_path)
        protocol_folds[subject_to_fold(subject_id)].append(sample_id_path)

    # Create the dataset metadata.
    metadata = {}
    metadata['name'] = args.name
    metadata['subjects'] = subject_list
    metadata['protocols'] = {'all': protocol_folds}

    # Write the metadata.json file.
    with open(output_dataset_meta, 'w') as meta_file:
        json.dump(metadata, meta_file)
