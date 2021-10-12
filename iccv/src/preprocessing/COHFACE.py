#!/usr/bin/env python3

import os
import sys
import pandas as pd
import json
import csv
import h5py
import cv2
import datetime
import dateutil

from src.preprocessing.preprocessing_utils import *
from src.shared.cv_utils import GetVideoMetadata


def Process_COHFACE(args):
    # Setup dataset constants
    kDatasetOrigin = datetime.date(2017, 9, 4)  # Date uploaded to arXiv.

    # Get the dataset metadata path.
    output_dataset_meta = os.path.join(args.out_dir, "metadata.json")

    # Loop through all samples.
    subject_list = []
    for subject_id in os.listdir(args.in_dir):
        if not IsInteger(subject_id):
            continue
        input_subject_path = os.path.join(args.in_dir, subject_id)
        output_subject_path = os.path.join(args.out_dir, subject_id)
        output_subject_meta = os.path.join(output_subject_path, "metadata.json")
        os.makedirs(output_subject_path, exist_ok = True)
        subject_age = 0
        subject_list.append(int(subject_id))
        sample_list = []
        for sample_id in os.listdir(input_subject_path):
            print("Processing subject", subject_id, "sample", sample_id)
            
            input_sample_path = os.path.join(input_subject_path, sample_id)
            output_sample_path = os.path.join(output_subject_path, sample_id)
            os.makedirs(output_sample_path, exist_ok = True)
            sample_list.append(int(sample_id))

            # Read the metadata.
            input_meta_path = os.path.join(input_sample_path, "data.hdf5")
            output_meta_path = os.path.join(output_sample_path, "metadata.json")
            output_phys_path = os.path.join(output_sample_path, "phys.csv")
            with h5py.File(input_meta_path, 'r') as hdf5_file:
                # Write the phys.csv file.
                Fs = float(hdf5_file.attrs['sample-rate-hz'][0])
                phys_data = {}
                phys_data['timestamp'] = hdf5_file['time'][:]
                phys_data['timestamp'] -= phys_data['timestamp'][0]
                phys_data['ppg'] = hdf5_file['pulse'][:]
                phys_data['respiration'] = hdf5_file['respiration'][:]
                df = pd.DataFrame(phys_data)
                df = CorrectIrregularlySampledData(df, 30.0)
                df.to_csv(output_phys_path)
                
                # Create the sample metadata.
                metadata = {}
                metadata['dataset'] = args.name
                metadata['subject'] = subject_id
                metadata['lighting'] = hdf5_file.attrs['illumination'][0].decode('utf-8')
                metadata['session'] = sample_id
                metadata['activity'] = "sitting"

                # Determine the video metadata.
                input_video_path = os.path.join(input_sample_path, "data.avi")
                video_metadata = GetVideoMetadata(input_video_path)
                relative_video_path = os.path.relpath(input_video_path, args.in_dir)
                video_metadata['path'] = os.path.join('data/raw/', args.name, relative_video_path)
                video_metadata['format'] = 'rgb'
                metadata['videos'] = {'main': video_metadata}

                # Determine the temporal metadata.
                phys_metadata = {}
                phys_metadata['path'] = "phys.csv"
                phys_metadata['total_entries'] = len(df['timestamp'])
                phys_metadata['sample_rate'] = 30.0
                metadata['temporal_data'] = {'phys': phys_metadata}

                # Determine minimum duration.
                skip_two_samples = 2.0 / phys_metadata['sample_rate']
                metadata['duration'] = min(phys_data['timestamp'][-1], video_metadata['temporal_data']['total_entries'] /
                                           video_metadata['temporal_data']['sample_rate']) - skip_two_samples
                metadata['timestamp_start'] = skip_two_samples

                # Write the metadata.json file.
                with open(output_meta_path, 'w') as meta_file:
                    json.dump(metadata, meta_file)

                # Determine subject age.
                birth_date = hdf5_file.attrs['birth-date'][0].decode('utf-8')
                birth_date = datetime.datetime.strptime(birth_date, '%d.%m.%Y').date()
                subject_age = int(dateutil.relativedelta.relativedelta(kDatasetOrigin, birth_date).years)

        # Create the subject metadata.
        metadata = {}
        metadata['dataset'] = args.name
        metadata['samples'] = sample_list
        metadata['samples'].sort()
        metadata['demographics'] = {'age': subject_age}

        # Write the metadata.json file.
        with open(output_subject_meta, 'w') as meta_file:
            json.dump(metadata, meta_file)

    # Load the protocol and fold files.
    protocols_path = os.path.join(args.in_dir, "protocols")
    protocols = LoadProtocols(protocols_path)

    # Create the dataset metadata.
    metadata = {}
    metadata['name'] = args.name
    metadata['subjects'] = subject_list
    metadata['subjects'].sort()
    metadata['protocols'] = protocols

    # Write the metadata.json file.
    with open(output_dataset_meta, 'w') as meta_file:
        json.dump(metadata, meta_file)
