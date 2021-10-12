#!/usr/bin/env python3

from itertools import chain
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.parse_args import parse_args
from src.setup_dataloader import setup_dataloader
from src.shared.torch_utils import set_random_seed


if __name__ == '__main__':
    # Parse args
    args = parse_args('stats')

    # Fix random seed for reproducability
    set_random_seed(args.random_seed)

    # Dataset and dataloader construction
    dataloader = setup_dataloader(args, 'stats')
    dataset = dataloader.dataset

    # Print dataset stats
    print()
    print('Dataset Name: {:s}'.format(dataset.dataset_metadata['name']))
    print('Total Subjects: {:d}'.format(len(dataset.dataset_metadata['subjects'])))
    print('Total Samples: {:d}'.format(len(dataset.samples)))
    subject_sample_numbers = [len(x) for x in dataset.subject_samples.values()]
    print('Samples Per Subject: {:.1f}±{:.1f}'.format(np.mean(subject_sample_numbers), np.std(subject_sample_numbers)))
    all_durations, all_hr = [], []
    video_res = {}
    for samples in dataloader:
        for sample_metadata in samples['sample_metadata']:
            for video_source, video_metadata in sample_metadata['videos'].items():
                video_res[video_source] = str(video_metadata['width']) + 'x' + str(video_metadata['height'])
            all_durations.append(sample_metadata['duration'])
            assert(len(samples.batch['targets']) == 3)
            all_hr.extend(samples.batch['targets']['rate'].tolist())
    print('Sample Duration: {:.1f}±{:.1f}'.format(np.mean(all_durations), np.std(all_durations)))
    print('Minimum Duration: {:.1f}'.format(np.min(all_durations)))
    for video_source, resolution in video_res.items():
        print(f'Video ({video_source}) Resolution: {resolution}')
    all_hr = list(chain(*all_hr))
    plt.hist(all_hr, 50)
    plt.title('ground truth HR')
    plt.savefig(f"figs/tmp_HR_hist_{dataset.dataset_metadata['name']}.png")
    plt.close()
