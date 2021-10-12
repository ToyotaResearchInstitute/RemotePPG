import sys
import shutil
import cv2
import numpy as np
import random
import json
import os
import pandas as pd
pd.options.mode.chained_assignment = None
import copy
from itertools import compress
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, lfilter
import glob
import csv

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision.transforms import RandomRotation, ToPILImage, ToTensor, ColorJitter, Resize
import torchvision.transforms.functional as TF
tr = torch

from src.shared.torch_utils import torch_random_uniform
from src.shared.heart_rate import predict_heart_rate
from src.losses.IrrelevantPowerRatio import IrrelevantPowerRatio
from src.shared.cv_utils import GetVideoMetadata, FrameReader
from src.shared.augmentation import ImageAugmentation


class RppgDataset(Dataset):
    def __init__(self, options):
        self.options = copy.deepcopy(options)

        # Overrides for first pass
        self.options.include_video = False
        self.options.include_metadata = True
        self.options.include_target = False
        self.options.chunk_dataset = False
        self.options.resample_bad_ppg = False

        # Open dataset metadata
        self.dataset_metadata = None
        if not self.options.unprocessed:
            dataset_metadata_path = os.path.join(self.options.preprocessed_path, "metadata.json")
            with open(dataset_metadata_path, 'r') as dataset_metadata_file:
                self.dataset_metadata = json.load(dataset_metadata_file)
            if self.dataset_metadata is None:
                raise Exception("Cannot read dataset metadata file: {:s}".format(dataset_metadata_path))
        else:
            # Search through dir for videos
            self.dataset_metadata = {}
            self.dataset_metadata['name'] = os.path.basename(self.options.dataset_path)
            subject_names = next(os.walk(self.options.dataset_path))[1]
            self.dataset_metadata['subjects'] = subject_names
        self.dataset_metadata['dataset_path'] = self.options.dataset_path
        self.dataset_metadata['preprocessed_path'] = self.options.preprocessed_path

        # Determine which folds to use
        if self.options.fold not in ['train', 'dev', 'test']:
            subset_folds = None
            if self.options.filter_presets is not None:
                subset_folds = self.options.filter_presets
        elif self.options.protocol == 'preset':
            subset_folds = [self.options.fold,]
        elif self.options.protocol == 'preset_add_test':
            if self.options.fold == 'train':
                subset_folds = ['train', 'test']
            else:
                subset_folds = [self.options.fold,]
        elif self.options.protocol == 'preset_mix_train_dev':
            if self.options.fold == 'train' or self.options.fold == 'dev':
                subset_folds = ['train', 'dev']
            else:
                subset_folds = [self.options.fold,]
        elif self.options.filter_presets is not None:
            subset_folds = self.options.filter_presets
        else:
            subset_folds = None

        self.temporal_split = 'none'
        if self.options.protocol == 'temporal':
            self.temporal_split = 'after' if self.options.fold == 'test' else 'before'

        # Open subject metadata
        self.samples = []
        self.subject_metadata_dict = {}
        use_all_samples = subset_folds is None or 'protocols' not in self.dataset_metadata or 'all' not in self.dataset_metadata['protocols']
        for subject in self.dataset_metadata['subjects']:
            subject_name = str(subject)
            self.subject_metadata_dict[subject_name] = None
            if not self.options.unprocessed:
                subject_metadata_path = os.path.join(self.options.preprocessed_path, subject_name, "metadata.json")
                with open(subject_metadata_path, 'r') as subject_metadata_file:
                    self.subject_metadata_dict[subject_name] = json.load(subject_metadata_file)
                if self.subject_metadata_dict[subject_name] is None:
                    raise Exception("Cannot read subject metadata file: {:s}".format(subject_metadata_path))
                if use_all_samples:
                    subject_sample_list = self.subject_metadata_dict[subject_name]['samples']
                    self.samples.extend([os.path.join(subject_name, str(x)) for x in subject_sample_list])
            else:
                subject_path = os.path.join(self.options.dataset_path, subject_name)
                sample_names = os.listdir(subject_path)
                self.subject_metadata_dict[subject_name] = {}
                self.subject_metadata_dict[subject_name]['samples'] = sample_names
                self.samples.extend([os.path.join(subject_name, str(x)) for x in sample_names])
        if not use_all_samples:
            for subset_fold in subset_folds:
                self.samples.extend(self.dataset_metadata['protocols']['all'][subset_fold])

        # Filter the samples for round robin protocol
        if 'round_robin' in self.options.protocol:
            train_with_dev = 'no_dev' in self.options.protocol

            # Determine which subject modulus to use
            keep_samples = []
            use_mods = set()
            if self.options.fold == 'test':
                use_mods.add(self.options.round_robin_index)
            elif self.options.fold == 'dev':
                use_mods.add((self.options.round_robin_index + 1) % 5)
            else:
                if train_with_dev:
                    use_mods.add((self.options.round_robin_index + 1) % 5)
                use_mods.add((self.options.round_robin_index + 2) % 5)
                use_mods.add((self.options.round_robin_index + 3) % 5)
                use_mods.add((self.options.round_robin_index + 4) % 5)

            # Filter for the correct subjects
            for sample in self.samples:
                subject_mod_id = int(sample.split('/')[0]) % 5
                if subject_mod_id in use_mods:
                    keep_samples.append(sample)
            self.samples = keep_samples

        # Filter subjects with ground truth errors from the UBFC test set
        if self.dataset_metadata['name'] == 'UBFC' and self.options.fold == 'test':
            keep_samples = []
            remove_subjects = set([11, 18, 20, 24])
            for sample in self.samples:
                subject_id = int(sample.split('/')[0])
                if subject_id not in remove_subjects:
                    keep_samples.append(sample)
            self.samples = keep_samples

        # Filter subjects with ground truth errors from the PURE test set
        if self.dataset_metadata['name'] == 'PURE' and self.options.fold == 'test':
            keep_samples = []
            remove_samples = set(['7/7', '7/2'])
            for sample in self.samples:
                if sample not in remove_samples:
                    keep_samples.append(sample)
            self.samples = keep_samples

        # Filter out invalid samples and excluded activities
        keep_samples = []
        source_durations = []
        for sample in self:
            if sample is None:
                continue
            if self.options.filter_activities is not None:
                if sample['sample_metadata']['activity'] not in self.options.filter_activities:
                    continue
            if self.options.filter_sessions is not None:
                if sample['sample_metadata']['session'] not in self.options.filter_sessions:
                    continue
            keep_samples.append(sample['id'])
            source_durations.append(sample['source_duration'])
        self.samples = keep_samples

        # Chunk the dataset into sections of length D
        self.options.chunk_dataset = options.chunk_dataset
        if self.options.chunk_dataset:
            chucking_offset = 0.0
            if self.options.protocol == 'temporal':
                chucking_offset = self.options.temporal_after_time
            new_samples = []
            self.sample_offsets = []
            for sample_id, duration in zip(self.samples, source_durations):
                num_chunks = int(np.floor((duration - chucking_offset) / self.options.sample_duration))
                for chunk_on in range(num_chunks):
                    new_samples.append(sample_id)
                    self.sample_offsets.append(chucking_offset + (chunk_on * self.options.sample_duration))
            self.samples = new_samples        

        # Randomly permute and downsample
        if self.options.downsample is not None and self.options.downsample != 1.0:
            random.shuffle(self.samples)
            upper_index = int(np.floor(len(self.samples) * self.options.downsample))
            self.samples = self.samples[:upper_index]

        # Create subject to sample map
        sample_set = set(self.samples)
        self.subject_samples = {}
        for subject in self.dataset_metadata['subjects']:
            subject_name = str(subject)
            for sample in self.subject_metadata_dict[subject_name]['samples']:
                sample_name = str(sample)
                if sample_name == "":
                    full_sample_name = subject_name
                else:
                    full_sample_name = os.path.join(subject_name, sample_name)
                if full_sample_name in sample_set:
                    if subject_name not in self.subject_samples:
                        self.subject_samples[subject_name] = []
                    self.subject_samples[subject_name].append(sample_name)

        # Undo include overrides
        self.options.include_video = options.include_video
        self.options.include_metadata = options.include_metadata
        self.options.include_target = options.include_target
        self.options.resample_bad_ppg = options.resample_bad_ppg

        # Setup bad PPG resampling
        if self.options.resample_bad_ppg:
            self.irrelevant_power_ratio = IrrelevantPowerRatio(self.options.target_freq, self.options.high_pass, self.options.low_pass)


    def __len__(self):
        return len(self.samples)


    # Determine path of data potentially relative to sample
    def parse_path(self, path, relative_root):
        if not os.path.exists(path):
            path = os.path.join(relative_root, path)
        return path


    # Reads temporal csv file and metadata
    def read_temporal_data(self, temporal_metadata, sample_path):
        if 'path' in temporal_metadata:
            temporal_path = self.parse_path(temporal_metadata['path'], sample_path)
            temporal_metadata['data'] = pd.read_csv(temporal_path)
        if 'timestamp_start' not in temporal_metadata:
            if 'data' in temporal_metadata and 'timestamp' in temporal_metadata['data']:
                temporal_metadata['timestamp_start'] = temporal_metadata['data']['timestamp'][0]
            else:
                temporal_metadata['timestamp_start'] = 0.0     


    # Determines the entry bounds within a temporal file (could be for targets or video frames)
    def get_temporal_bounds(self, temporal_metadata, sample_time_offset, sample_unscaled_duration, use_exact=False):
        if use_exact and 'data' in temporal_metadata and 'timestamp' in temporal_metadata['data']:
            timestamps = temporal_metadata['data']['timestamp'].to_numpy()
            temporal_start_time = sample_time_offset - temporal_metadata['timestamp_start'] - (1 / temporal_metadata['sample_rate'])
            temporal_end_time = temporal_start_time + sample_unscaled_duration + (1 / temporal_metadata['sample_rate'])
            temporal_filter = np.logical_and(timestamps >= temporal_start_time, timestamps < temporal_end_time)
            subset_indices = np.where(temporal_filter)[0]
            return (subset_indices[0], subset_indices[-1] + 1)
        else:
            temporal_start_time = sample_time_offset - temporal_metadata['timestamp_start']
            temporal_number_entries = int(np.round(sample_unscaled_duration * temporal_metadata['sample_rate']))
            temporal_start_entry = int(np.round(temporal_start_time * temporal_metadata['sample_rate']))
            temporal_end_entry = temporal_start_entry + temporal_number_entries
            return (temporal_start_entry, temporal_end_entry)
    
    
    def __getitem__(self, sample_idx):
        if torch.is_tensor(sample_idx):
            sample_idx = sample_idx.tolist()

        # Set up ppg shift injection
        if self.options.inject_ppg_shift > 0:
            ppg_shift = torch_random_uniform(-self.options.inject_ppg_shift, self.options.inject_ppg_shift).item()
        else:
            ppg_shift = 0

        # Read the sample metadata
        sample_metadata = None
        sample_id = self.samples[sample_idx]
        sample_path = os.path.join(self.options.preprocessed_path, sample_id)            
        if not self.options.unprocessed:
            sample_metadata_path = os.path.join(sample_path, "metadata.json")
            if not os.path.isfile(sample_metadata_path):
                return None
            with open(sample_metadata_path, 'r') as sample_metadata_file:
                sample_metadata = json.load(sample_metadata_file)
            if sample_metadata is None:
                raise Exception("Cannot read sample metadata file: {:s}".format(sample_metadata_path))
        else:
            video_path = os.path.join(self.options.dataset_path, sample_id)
            sample_metadata = {}
            sample_metadata['subject'] = sample_id.split('/')[0]
            sample_metadata['videos'] = {}
            sample_metadata['videos']['main'] = GetVideoMetadata(video_path)
            sample_metadata['videos']['main']['path'] = video_path
        sample_metadata['id'] = sample_id
        sample_metadata['sample_path'] = sample_path

        # Check for missing video file
        if 'main' not in sample_metadata['videos']:
            return None
        if sample_metadata['videos']['main']['temporal_data']['total_entries'] == 0:
            return None

        # Populate default duration
        if 'duration' not in sample_metadata:
            sample_metadata['duration'] = sample_metadata['videos']['main']['temporal_data']['total_entries'] / sample_metadata['videos']['main']['temporal_data']['sample_rate']

        # Ignore samples which are too short
        if self.options.sample_duration is not None:
            min_factor = 1.0 if self.options.augment_freq_range is None else self.options.augment_freq_range[0]
            if sample_metadata['duration'] < self.options.sample_duration * min_factor:
                return None

        # Read the subject metadata
        subject_metadata = self.subject_metadata_dict[str(sample_metadata['subject'])]

        # Read test subject ID, if exists
        test_subject_id_path = os.path.join(sample_path, 'main_subject_id.txt')
        if os.path.isfile(test_subject_id_path):
            with open(test_subject_id_path, 'r') as id_file:
                sample_metadata['test_subject_id'] = int(id_file.readline())

        # Form sample
        sample = {}
        sample['id'] = sample_id
        sample['source_duration'] = sample_metadata['duration']           
        if self.options.include_metadata:
            sample['dataset_metadata'] = self.dataset_metadata
            sample['subject_metadata'] = subject_metadata
            sample['sample_metadata'] = sample_metadata

        # Load the phys target
        if self.options.include_target:
            # Repeat target loading until good PPG found
            current_power_thresh = 0.6 # discards between 0.1-2% of supervised samples
            good_ppg_found = False
            while not good_ppg_found:
                # Set up frequency augmentation
                if self.options.augment_freq_range is None:
                    freq_scale_fact = 1.0
                else:
                    freq_scale_fact = torch_random_uniform(self.options.augment_freq_range[0], self.options.augment_freq_range[1])

                # Determine sample offset
                if self.options.sample_duration is None:
                    sample_time_offset = 0.0
                    sample_unscaled_duration = sample_metadata['duration']
                    if self.temporal_split == 'before':
                        sample_unscaled_duration = self.options.temporal_before_time
                    elif self.temporal_split == 'after':
                        sample_unscaled_duration -= self.options.temporal_after_time
                        sample_time_offset = self.options.temporal_after_time
                    sample_scaled_d = int(np.floor(sample_unscaled_duration * self.options.target_freq / freq_scale_fact))
                else:
                    sample_unscaled_duration = self.options.sample_duration * freq_scale_fact
                    sample_scaled_d = self.options.D
                    if self.options.chunk_dataset:
                        sample_time_offset = self.sample_offsets[sample_idx]
                    else:
                        if self.options.percent_offset is None:
                            percent_offset = torch.rand(1)
                        else:
                            percent_offset = self.options.percent_offset
                        if self.temporal_split == 'before':
                            sample_time_offset = percent_offset * (self.options.temporal_before_time - sample_unscaled_duration)
                        elif self.temporal_split == 'after':
                            sample_time_offset = percent_offset * (sample_metadata['duration'] - self.options.temporal_after_time - sample_unscaled_duration)
                        else:
                            sample_time_offset = percent_offset * (sample_metadata['duration'] - sample_unscaled_duration)
                sample_id_time = sample_id + "-{:d}".format(int(np.round(sample_time_offset)))
                if 'timestamp_start' in sample_metadata:
                    sample_time_offset += sample_metadata['timestamp_start']

                # Check that offset and duration within bounds of sample
                if sample_time_offset < 0. or sample_time_offset + sample_unscaled_duration > sample_metadata['duration']:
                    continue

                # Determine label names based on dataset
                if 'ECG' in self.options.preprocessed_path:
                    in_label_names = ('ecg', 'ecg_heart_rate')
                elif 'UBFC-Phys' in self.options.preprocessed_path:
                    in_label_names = ('bvp', 'bvp_heart_rate')
                else:
                    in_label_names = ('ppg', 'ppg_heart_rate')
                out_label_names = ('ppg', 'rate')

                # Read the temporal data
                out_of_bounds = False
                temporal_data = {}
                if 'temporal_data' in sample_metadata:
                    use_dummy_hr = False
                    for temporal_name, temporal_metadata in sample_metadata['temporal_data'].items():
                        self.read_temporal_data(temporal_metadata, sample_path)
                        temporal_start_entry, temporal_end_entry = self.get_temporal_bounds(temporal_metadata, ppg_shift + sample_time_offset, sample_unscaled_duration, use_exact=False)
                        df = temporal_metadata['data']
                        if temporal_start_entry < 0 or temporal_end_entry > len(df):
                            out_of_bounds = True
                            break
                        temporal_metadata['data'] = df[temporal_start_entry:temporal_end_entry]
                        temporal_data[temporal_name] = temporal_metadata
                    if out_of_bounds:
                        continue  # Resample
                else:
                    # Create dummy data
                    use_dummy_hr = True
                    temporal_data['phys'] = {}
                    temporal_data['phys']['total_entries'] = sample_metadata['videos']['main']['temporal_data']['total_entries']
                    temporal_data['phys']['sample_rate'] = sample_metadata['videos']['main']['temporal_data']['sample_rate']
                    temporal_data['phys']['data'] = {}
                    temporal_rows = {}
                    for name in in_label_names:
                        temporal_rows[name] = np.zeros((temporal_data['phys']['total_entries'],))
                    temporal_data['phys']['data'] = pd.DataFrame.from_dict(temporal_rows)

                # Construct target signals
                targets = {}
                for idx, label_name in enumerate(in_label_names):
                    if label_name in temporal_data['phys']['data']:
                        new_signal = temporal_data['phys']['data'][label_name].to_numpy()
                        new_signal[np.isnan(new_signal)] = 0.0
                        targets[out_label_names[idx]] = new_signal

                # Create dummy PPG data, if missing
                if 'ppg' not in targets:
                    targets['ppg'] = np.zeros((temporal_data['phys']['total_entries'],))

                # Calculate heart rate from PPG
                targets['our'] = predict_heart_rate(targets['ppg'], temporal_data['phys']['sample_rate'], self.options.high_pass, self.options.low_pass)
                targets['our'] = np.repeat(targets['our'], len(targets['ppg']))
                if 'rate' not in targets:
                    targets['rate'] = targets['our']

                # Move targets to tensors
                for key, target in targets.items():
                    targets[key] = tr.from_numpy(target).type(tr.FloatTensor)

                # Frequency augment target signals
                targets = self.freq_augm_targets(targets, sample_scaled_d, freq_scale_fact)

                # Check for HR outside allowable range
                if not use_dummy_hr and self.options.augment_freq_range is not None:
                    if targets['rate'][0] < self.options.high_pass or targets['rate'][0] > self.options.low_pass:
                        continue  # Resample

                # Normalize PPG signals
                cur_mean = torch.mean(targets['ppg'])
                cur_std = torch.std(targets['ppg'])
                targets['ppg'] = (targets['ppg'] - cur_mean) / cur_std
                not_outlier = (np.abs(targets['ppg']) < 4)
                cur_mean = torch.mean(targets['ppg'][not_outlier])
                cur_std = torch.std(targets['ppg'][not_outlier])
                targets['ppg'] = (targets['ppg'] - cur_mean) / cur_std

                # Check if bad PPG - resample if so
                if self.options.resample_bad_ppg:
                    ratio = self.irrelevant_power_ratio(targets['ppg'][None,:], None).item()
                    good_ppg_found = ratio < current_power_thresh
                    if not good_ppg_found:
                        print("WARNING: Resampling bad PPG")
                        current_power_thresh = (0.9 * current_power_thresh) + 0.1
                else:
                    good_ppg_found = True

            sample['targets'] = targets
            sample['id_time'] = sample_id_time
            sample['length'] = len(targets['ppg'])

        # Read video data
        video_filename = os.path.split(sample_metadata['videos']['main']['path'])[1]
        sample['video_filename'] = video_filename
        if self.options.include_video:
            if self.options.video_source == 'dummy':
                video = tr.rand((self.options.C, video_end_frame - video_start_frame, self.options.crop_height, self.options.crop_width))           
            else:
                # Read raw video metadata
                raw_video_type = self.options.video_source.split('_')[0]
                video_metadata = sample_metadata['videos'][raw_video_type]
                self.read_temporal_data(video_metadata['temporal_data'], sample_path)
                video_start_frame, video_end_frame = self.get_temporal_bounds(video_metadata['temporal_data'], sample_time_offset, sample_unscaled_duration)
                if raw_video_type != self.options.video_source:
                    video_path = self.parse_path(self.options.video_source, sample_path)
                    video_format = 'rgb'
                else:
                    video_path = self.parse_path(video_metadata['path'], sample_path)
                    video_format = video_metadata['format'] if 'format' in video_metadata else 'rgb'

                # Setup video cropping
                if self.options.crop_height is not None:
                    frame_height = self.options.crop_height
                    frame_width = self.options.crop_width
                else:
                    frame_width = video_metadata['height'] * self.options.crop_aspect_ratio
                    frame_width = min(frame_width, video_metadata['width'])
                    frame_height = int(frame_width / self.options.crop_aspect_ratio)
                    frame_width = int(frame_width)

                # Setup image augmentation
                if self.options.augment_image:
                    image_augmentor = ImageAugmentation(frame_height, frame_width, W=1)

                # Read the video data
                video = tr.empty(self.options.C, video_end_frame - video_start_frame, frame_height, frame_width, dtype=tr.float)
                for frame_idx, img in enumerate(FrameReader(video_path, video_start_frame, video_end_frame, video_format, target_format='rgb')):
                    # Convert to tensor
                    out_type = img.dtype
                    out_type_info = np.iinfo(out_type)
                    img = tr.from_numpy(img.astype(np.float32)).permute(2,0,1)

                    # Normalize between 0 and 1
                    img_range = out_type_info.max - out_type_info.min
                    img = (img - out_type_info.min) / img_range
                    
                    # Crop to desired spatial resolution
                    height_offset = int((img.shape[-2] - frame_height) / 2)
                    width_offset = int((img.shape[-1] - frame_width) / 2)
                    img = img[:,height_offset:height_offset+frame_height,width_offset:width_offset+frame_width]

                    # Take channel subset
                    img = img[self.options.channel_idx]

                    # Augment image
                    if self.options.augment_image:
                        img = image_augmentor(img)

                    # Output frame
                    video[:, frame_idx] = img

            # Frequency augment and spatial scaling
            video = self.scale_video(video, sample_scaled_d, self.options.scale_height, self.options.scale_width)

            # Periodic noise injection experiment
            if self.options.noise_block is not None:
                # keep simple for the moment
                # video is 3x300x64x64
                vid_c, vid_d, vid_h, vid_w = video.shape

                buffer = 5
                duty = self.options.noise_duty
                assert(0 < duty < 1)
                width = self.options.noise_size
                assert((buffer + width) < vid_w)

                assert(0 < len(self.options.noise_period) <= 2)
                if len(self.options.noise_period) == 2:
                    period_min = self.options.noise_period[0]
                    period_max = self.options.noise_period[1]
                    period = np.random.randint(period_min, period_max)
                else:
                    period = self.options.noise_period[0]

                if self.options.noise_block == 1:
                    # apply red block (3 channel) or white block (1 channel) in top left
                    select_c = 0
                    x1 = buffer
                    x2 = buffer + width
                    offset = np.random.randint(0, period)
                    ind = np.arange(0, vid_d, 1) + offset
                    duty_period = int(period * duty)
                    noise_on_ind = (ind % period) < duty_period
                    video[0, noise_on_ind, x1:x2, x1:x2] = 1
                else:
                    raise NotImplementedError

            sample['video'] = video  # Video shape: C x D x H X W

        return sample


    # Custom collation used when collecting metadata
    class CustomBatch:
        def collate_list(self, samples):
            sizes = [x.shape for x in samples]
            max_size = tuple(np.stack(sizes).max(axis=0))
            stacked = torch.empty((len(samples),) + max_size)
            for i, sample in enumerate(samples):
                # Iterate through sizes backwards, as pad expects them in reverse order.
                padding_amount = []
                for cur_size_dim, max_size_dim in zip(np.flip(sizes[i]), reversed(max_size)):
                    padding_amount.append(0)
                    padding_amount.append(max_size_dim - cur_size_dim)
                stacked[i] = F.pad(sample, padding_amount)
            return stacked

        def __init__(self, samples):
            self.batch = {}
            for key in samples[0].keys():
                if key == 'targets':
                    self.batch[key] = {}
                    for target_key in samples[0]['targets'].keys():
                        self.batch[key][target_key] = self.collate_list([sample[key][target_key] for sample in samples])
                elif key == 'video':
                    self.batch[key] = self.collate_list([sample[key] for sample in samples])
                else:
                    self.batch[key] = []
                    for i in range(len(samples)):
                        self.batch[key].append(samples[i][key])

        # custom memory pinning method on custom type
        def pin_memory(self):
            if 'video' in self.batch:
                self.batch['video'] = self.batch['video'].pin_memory()
            if 'targets' in self.batch:
                for target_key in self.batch['targets'].keys(): 
                    self.batch['targets'][target_key] = self.batch['targets'][target_key].pin_memory()
            return self

        def __getitem__(self, key):
            return self.batch[key]

        def __setitem__(self, key, value):
            self.batch[key] = value


    @staticmethod
    def collate_fn(samples):
        return RppgDataset.CustomBatch(samples)


    # Frequency augmentation for targets
    def freq_augm_targets(self, targets, D, freq_scale_fact):
        resampler = torch.nn.Upsample(size=(D,), mode='linear', align_corners=False)
        for key, target in targets.items():
            if key in ['rate', 'our']:
                target = target * freq_scale_fact
            targets[key] = resampler(target.view(1, 1, -1)).squeeze(1).squeeze(0)
        return targets


    # Frequency augmentation for video
    def scale_video(self, video, D, H, W):
        resampler = torch.nn.Upsample(size=(D, H, W), mode='trilinear', align_corners=False)
        return resampler(video.unsqueeze(0)).squeeze(0)
