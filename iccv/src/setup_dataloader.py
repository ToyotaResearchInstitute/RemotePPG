import os
import numpy as np

import torch
from torch.utils.data import DataLoader, ConcatDataset, RandomSampler

from src.dset import RppgDataset
from src.model_loader import load_model


videoChannelNameIndexMap = {'red': 0,
                            'green': 1,
                            'blue': 2,
                            'bright': 0,
                            'dark': 1,
                            'subtract': 2,
                            'Y': 0,
                            'U': 1,
                            'V': 2}

class DatasetOptions:
    def __init__(self, args, mode, augment):
        # Setup dataset path
        self.dataset_path = None  # Overriden below
        self.preprocessed_path = None  # Overriden below
        self.unprocessed = args.unprocessed
        self.ground_truth_model = None  # Overriden in training script when using

        # Determine loader device
        if args.n_cpu == 0:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        # Setup chunking
        if mode == 'train' or mode == 'dev':
            self.chunk_dataset = False
        else:
            self.chunk_dataset = args.test_chunk_dataset

        # Noise injection
        self.noise_block = args.noise_block if mode != 'stats' else None
        self.noise_size = args.noise_size if mode != 'stats' else None
        self.noise_period = args.noise_period if mode != 'stats' else None
        self.noise_duty = args.noise_duty if mode != 'stats' else None

        # Protocol
        self.protocol = args.protocol
        self.round_robin_index = args.round_robin_index
        self.temporal_before_time = args.temporal_before_time
        self.temporal_after_time = args.temporal_after_time
        self.fold = mode

        # Feature selection
        self.video_source = args.video_source
        self.D = args.time_depth
        self.C = len(args.use_channels)
        self.H = args.scale_height
        self.W = args.scale_width
        self.aspect_ratio = self.W / self.H
        self.channel_idx = np.array([videoChannelNameIndexMap[x] for x in args.use_channels])
        self.crop_aspect_ratio = args.crop_aspect_ratio
        self.crop_height = args.crop_height
        self.crop_width = args.crop_width
        self.scale_height = args.scale_height
        self.scale_width = args.scale_width
        self.percent_offset = (None if augment else 0.5)

        # Frame rate
        self.target_freq = 30.
        self.sample_duration = None if self.D == 0 else self.D / self.target_freq

        # Filtering samples
        self.resample_bad_ppg = False if mode == 'test' else args.resample_bad_ppg
        self.downsample = args.dev_downsample if mode == 'dev' else args.downsample
        self.filter_activities = args.filter_activities
        self.filter_sessions = args.dev_filter_sessions if mode == 'dev' else args.filter_sessions
        self.filter_presets = args.filter_presets
    
        # Sample augmentation
        self.inject_ppg_shift = 0 if mode == 'test' else args.inject_ppg_shift
        self.augment_freq_range = (args.freq_augm_range if augment else None)
        self.augment_image = (args.img_augm if augment else False)

        # Modality selection
        self.include_video = mode != 'stats' and not args.skip_videos
        self.include_metadata = (mode == 'stats')
        self.include_target = True

        # Filter
        self.low_pass = args.low_pass
        self.high_pass = args.high_pass


def setup_dataloader(args, mode):
    # Setup dataset options
    augment = (mode == 'train')
    dataset_options = DatasetOptions(args, mode, augment)

    # Initialize datasets
    datasets = []
    for dataset_name in args.datasets:
        dataset_options.dataset_path = os.path.join('data/raw', dataset_name)
        dataset_options.preprocessed_path = os.path.join('data/preprocessed', dataset_name)
        datasets.append(RppgDataset(dataset_options))

    # Combine datasets
    dataset = ConcatDataset(datasets)
    dataset.options = datasets[0].options
    dataset.collate_fn = datasets[0].collate_fn
    dataset.dataset_metadata = datasets[0].dataset_metadata
    dataset.samples = datasets[0].samples
    dataset.subject_samples = datasets[0].subject_samples

    # Construct DataLoader
    sampler = None
    if augment and args.train_sampler_number is not None:
        sampler = RandomSampler(dataset, replacement=True, num_samples=args.train_sampler_number)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=(augment and sampler is None),
                            num_workers=args.n_cpu,
                            pin_memory=True,
                            sampler=sampler,
                            collate_fn=None if mode in ['train', 'dev'] else dataset.collate_fn)

    mode_name = '' if mode is None else mode + ' '
    print('Created {:s}dataset with {:d} samples'.format(mode_name, len(dataset)))

    return dataloader
