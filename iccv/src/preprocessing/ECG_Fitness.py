#!/usr/bin/env python3

import os
import sys
import pandas as pd
import json
import csv
import cv2
import datetime
import dateutil
from zipfile38 import ZipFile
import glob

from src.preprocessing.preprocessing_utils import *
from src.shared.cv_utils import GetVideoMetadata


def convert_timestamps(current_timestamps, first_timestamp):
    return (current_timestamps - first_timestamp) * 1e-3

def Process_ECG_Fitness(args):
    # Setup dataset constants
    lighting_map = {}
    lighting_map[1] = 'natural'
    lighting_map[2] = 'natural'
    lighting_map[3] = 'lamp'
    lighting_map[4] = 'lamp'
    lighting_map[5] = 'natural'
    lighting_map[6] = 'natural'

    activity_map = {}
    activity_map[1] = 'talking'
    activity_map[2] = 'rowing'
    activity_map[3] = 'talking'
    activity_map[4] = 'rowing'
    activity_map[5] = 'elliptical'
    activity_map[6] = 'bike'

    remove_video_list = [' PPG', ' PPG HR', ' SpO2', ' PI']
    rename_video_map = {}
    rename_video_map['milliseconds'] = 'timestamp'
    rename_video_map[' ECG'] = 'ecg'
    rename_video_map[' ECG HR'] = 'ecg_heart_rate'

    rename_bbox_map = {}
    rename_bbox_map[1] = 'face_bbox_x'
    rename_bbox_map[2] = 'face_bbox_y'
    rename_bbox_map[3] = 'face_bbox_width'
    rename_bbox_map[4] = 'face_bbox_height'

    # Get the dataset metadata path.
    output_dataset_meta = os.path.join(args.out_dir, "metadata.json")

    # Loop through all sample json.
    input_dataset_bbox_dir = os.path.join(args.in_dir, 'bbox')
    subject_list = list(range(17))
    for subject_id in subject_list:
        input_subject_path = os.path.join(args.in_dir, '{:02d}'.format(subject_id))
        input_subject_bbox_path = os.path.join(input_dataset_bbox_dir, '{:02d}'.format(subject_id))
        output_subject_path = os.path.join(args.out_dir, str(subject_id))
        output_subject_meta = os.path.join(output_subject_path, "metadata.json")
        os.makedirs(output_subject_path, exist_ok = True)
        if subject_id == 2:
            sample_list = [1, 2, 3, 5, 6]  # Subject 2 is missing the fourth session.
        elif subject_id == 7:
            sample_list = [1, 3, 4, 5, 6]  # Subject 7 is missing the second session.
        else:
            sample_list = list(range(1,7))
        for sample_id in sample_list:
            print("Processing subject", subject_id, "sample", sample_id)

            # Get input paths.
            input_sample_path = os.path.join(input_subject_path, '{:02d}'.format(sample_id))
            input_sample_bbox_path = os.path.join(input_subject_bbox_path, '{:02d}'.format(sample_id))
            input_side_video_path = os.path.join(input_sample_path, 'c920-1.avi')
            input_machine_video_path = os.path.join(input_sample_path, 'c920-2.avi')
            input_video_frame_path  = os.path.join(input_sample_path, 'c920.csv')
            input_phys_path  = os.path.join(input_sample_path, 'viatom-raw.csv')
            input_side_bbox_path = os.path.join(input_sample_bbox_path, 'c920-1.face')
            input_machine_bbox_path = os.path.join(input_sample_bbox_path, 'c920-2.face')

            # Get output paths.
            output_sample_path = os.path.join(output_subject_path, str(sample_id))
            os.makedirs(output_sample_path, exist_ok = True)
            output_meta_path = os.path.join(output_sample_path, "metadata.json")
            output_side_path = os.path.join(output_sample_path, "side.csv")
            output_main_path = os.path.join(output_sample_path, "main.csv")
            output_phys_path = os.path.join(output_sample_path, "phys.csv")

            # Subject 11 has sample 1 and 2 swapped
            corrected_sample_id = sample_id
            if subject_id == 11:
                if sample_id == 1:
                    corrected_sample_id = 2
                elif sample_id == 2:
                    corrected_sample_id = 1

            # Create the sample metadata.
            metadata = {}
            metadata['dataset'] = args.name
            metadata['subject'] = subject_id
            metadata['lighting'] = lighting_map[corrected_sample_id]
            metadata['session'] = sample_id
            metadata['activity'] = activity_map[corrected_sample_id]

            # Read and write the phys data.
            phys_data = pd.read_csv(input_phys_path)
            phys_data = phys_data.drop(columns = remove_video_list, errors="raise")
            phys_data = phys_data.rename(columns = rename_video_map, errors="raise")
            first_timestamp = phys_data['timestamp'][0]
            phys_data['timestamp'] = convert_timestamps(phys_data['timestamp'], first_timestamp)
            nan_indices = np.where(np.isnan(phys_data['ecg']))[0]
            if len(nan_indices) > 0:
                phys_data = phys_data[:nan_indices[0]]
            where_negative = phys_data['ecg_heart_rate'] < 0
            phys_data.loc[where_negative, 'ecg_heart_rate'] = phys_data['ecg_heart_rate'][where_negative] + 256
            df = CorrectIrregularlySampledData(phys_data, 30.0)
            df.to_csv(output_phys_path)

            # Determine the temporal metadata.
            phys_metadata = {}
            phys_metadata['path'] = "phys.csv"
            phys_metadata['total_entries'] = len(df['timestamp'])
            phys_metadata['sample_rate'] = 30.0
            metadata['temporal_data'] = {'phys': phys_metadata}

            # Read frame timestamps.
            frame_index_data = pd.read_csv(input_video_frame_path, header=None)
            frame_index_data = frame_index_data.drop(columns = [0,], errors="raise")
            frame_index_data = frame_index_data.rename(columns = {1: 'index'}, errors="raise")
            frame_timestamps = []
            for _, row in frame_index_data.iterrows():
                if row['index'] >= len(phys_data['timestamp']):
                    break
                frame_timestamps.append(phys_data['timestamp'][row['index']])

            # Write side video csv.
            side_data = pd.read_csv(input_side_bbox_path, header=None, sep=' ')
            side_data = side_data.drop(range(len(frame_timestamps), len(side_data)))
            side_data = side_data.drop(columns = [0,], errors="raise")
            side_data = side_data.rename(columns = rename_bbox_map, errors="raise")
            side_data.insert(0, 'timestamp', frame_timestamps)
            side_data.to_csv(output_side_path)

            # Write main (machine) video csv.
            machine_data = pd.read_csv(input_machine_bbox_path, header=None, sep=' ')
            machine_data = machine_data.drop(range(len(frame_timestamps), len(machine_data)))
            machine_data = machine_data.drop(columns = [0,], errors="raise")
            machine_data = machine_data.rename(columns = rename_bbox_map, errors="raise")
            machine_data.insert(0, 'timestamp', frame_timestamps)
            machine_data.to_csv(output_main_path)

            # Determine the side video metadata.
            side_video_metadata = GetVideoMetadata(input_side_video_path)
            relative_side_video_path = os.path.relpath(input_side_video_path, args.in_dir)
            side_video_metadata['path'] = os.path.join('data/raw/', args.name, relative_side_video_path)
            side_video_metadata['temporal_data']['path'] = "side.csv"
            side_video_metadata['format'] = 'rgb'
            
            # Determine the main (machine) video metadata.
            machine_video_metadata = GetVideoMetadata(input_machine_video_path)
            relative_machine_video_path = os.path.relpath(input_machine_video_path, args.in_dir)
            machine_video_metadata['path'] = os.path.join('data/raw/', args.name, relative_machine_video_path)
            machine_video_metadata['temporal_data']['path'] = "main.csv"
            machine_video_metadata['format'] = 'rgb'
            
            metadata['videos'] = {'main': machine_video_metadata, 'side': side_video_metadata}

            # Determine minimum duration.
            metadata['timestamp_start'] = max(phys_data.iloc[0]['timestamp'], frame_timestamps[0])
            metadata['duration'] = min(phys_data.iloc[-1]['timestamp'], frame_timestamps[-1]) - metadata['timestamp_start']

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
