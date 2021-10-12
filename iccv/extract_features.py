#!/usr/bin/env python3

import os
import numpy as np

import torch
from torch.multiprocessing import Pool, set_start_method, Manager

from src.parse_args import parse_args
from src.setup_dataloader import setup_dataloader
from src.feature_extractor import FeatureExtractor
from src.shared.torch_utils import set_random_seed


if __name__ == '__main__':
    # Parse args
    args = parse_args('features')

    # Fix random seed for reproducability
    set_random_seed(args.random_seed)

    # Setup multiprocessing
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    n_devices = torch.cuda.device_count()
    manager = Manager()
    n_workers = args.n_cpu if args.n_cpu > 0 else 1
    available_gpus = manager.list(['cuda:' + str(x % n_devices) for x in list(range(n_workers))])

    # Dataset and dataloader construction
    dataloader = setup_dataloader(args, 'stats')
    dataset = dataloader.dataset

    # Setup feature extractor
    options = {}
    options['W'] = 128
    options['H'] = 192
    options['aspect_ratio'] = options['W'] / options['H']
    options['crop_overwrite'] = args.crop_overwrite
    options['crop_method'] = 's3fd'
    options['crop_border'] = [0.125, 0.125]
    options['crop_deadzone'] = 0.125
    options['crop_smooth_time'] = 0.25
    options['video_source'] = args.video_source
    options['available_gpus'] = available_gpus
    feature_extractor = FeatureExtractor(options)

    # Extract all features
    if args.n_cpu > 0:
        results = []
        pool=Pool(processes=args.n_cpu)
        for samples in dataloader:
            for metadata in samples['sample_metadata']:
                r = pool.apply_async(feature_extractor.extract_sample, args=(metadata,))
                results.append(r)

        pool.close()
        for r in results:
            r.wait()
            if not r.successful():
                # Raises an error when not successful
                r.get()

        pool.join()
    else:
        for samples in dataloader:
            for metadata in samples['sample_metadata']:
                feature_extractor.extract_sample(metadata)
