#!/usr/bin/env python3

import os
import cv2
import numpy as np
import pandas as pd


def IsInteger(text):
    try: 
        int(text)
        return True
    except ValueError:
        return False


def LoadProtocols(protocols_path):
    protocols = {}
    for protocol_name in os.listdir(protocols_path):
        protocols[protocol_name] = {}
        protocol_path = os.path.join(protocols_path, protocol_name)
        for fold_filename in os.listdir(protocol_path):
            fold_name = os.path.splitext(fold_filename)[0]
            protocols[protocol_name][fold_name] = []
            fold_path = os.path.join(protocol_path, fold_filename)
            with open(fold_path, "r") as fold_file:
                for sample_path in fold_file:
                    sample_parts = os.path.split(os.path.split(sample_path.strip())[0])
                    if len(sample_parts[0]) == 0:
                        sample_parts = sample_parts[1].split('-')
                    if len(sample_parts) != 2:
                        raise Exception('Protocol sample formatted incorrectly in file: {:s}'.format(sample_path))
                    sample_parts = [str(int(x)) for x in sample_parts]
                    sample_name = '/'.join(sample_parts)
                    protocols[protocol_name][fold_name].append(sample_name)
            protocols[protocol_name][fold_name] = list(np.unique(protocols[protocol_name][fold_name]))
    return protocols


def CorrectIrregularlySampledData(df, Fs):
    if df.iloc[0]['timestamp'] > 0.0:
        top_row = df.iloc[[0]].copy()
        df = pd.concat([top_row, df], ignore_index=True)
        df.loc[0, 'timestamp'] = 0.0
    new_data = []
    for frame_on, time_on in enumerate(np.arange(0.0, df.iloc[-1]['timestamp'], 1 / Fs)):
        time_diff = (df['timestamp'] - time_on).to_numpy()
        stop_idx = np.argmax(time_diff > 0)
        start_idx = stop_idx - 1
        time_span = time_diff[stop_idx] - time_diff[start_idx]
        rel_time = -time_diff[start_idx]
        stop_weight = rel_time / time_span
        start_weight = 1 - stop_weight
        average_row = pd.concat([df.iloc[[start_idx]].copy() * start_weight, df.iloc[[stop_idx]].copy() * stop_weight]).sum().to_frame().T
        new_data.append(average_row)
    return pd.concat(new_data)
