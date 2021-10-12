import os
import sys
import dlib
import cv2
import shutil
import numpy as np
from PIL import Image, ImageDraw
from pyDOE2 import ff2n
import random
import copy
import csv
import copy
import traceback
import face_recognition

import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor
tr = torch

from src.face_tracker.track_face import track_face_s3fd
from src.shared.bbox import bbox_add_border, bbox_inside_bbox, bbox_expand_aspect_ratio, BboxSmoother
from src.shared.cv_utils import img2uint8, FrameReader, VideoWriterContext


class FeatureExtractor:
    def __init__(self, options):
        self.options = options
        self.crop_dataset_workspace = {}
        self.landmark_dataset_workspace = {}
        self.tar_aspect_ratio = float(self.options['W']) / float(self.options['H'])

    # Convert the bbox to a subbset grid for use with grid_sample
    def generate_subset_grid(self, bbox, original_width, original_height):
        bbox = copy.deepcopy(bbox)
        bbox_width = bbox['x2'] - bbox['x1']
        bbox_height = bbox['y2'] - bbox['y1']
        bbox_aspect_ratio = bbox_width / bbox_height
        if bbox_aspect_ratio > self.options['aspect_ratio']:
            # Wider than needs to be - expand height
            new_height = bbox_width / self.options['aspect_ratio']
            delta_y = (new_height - bbox_height) / 2
            bbox['y1'] -= delta_y
            bbox_height = new_height
        else:
            # Higher than needs to be - expand width
            new_width = bbox_height * self.options['aspect_ratio']
            delta_x = (new_width - bbox_width) / 2
            bbox['x1'] -= delta_x
            bbox_width = new_width
        bbox['x1'] = (2 * bbox['x1'] / original_width) - 1
        bbox['x2'] = (2 * bbox['x2'] / original_width) - 1
        x_lin = tr.linspace(bbox['x1'], bbox['x2'], self.options['W'])
        bbox['y1'] = (2 * bbox['y1'] / original_height) - 1
        bbox['y2'] = (2 * bbox['y2'] / original_height) - 1
        y_lin = tr.linspace(bbox['y1'], bbox['y2'], self.options['H'])
        meshx, meshy = tr.meshgrid([y_lin, x_lin])
        return tr.stack((meshy, meshx), 2)[None, :]

    # Crop a frame of the video, using the designated crop_method
    def crop_frame(self, img, video_metadata, frame_idx, sample_workspace, device='cuda'):
        original_height = img.shape[0]
        original_width = img.shape[1]

        # Setup sample workspace
        for key in ['subset_grid', 'detected_bbox', 'border_bbox', 'target_bbox', 'final_bbox']:
            if key not in sample_workspace:
                sample_workspace[key] = None
        if self.options['crop_smooth_time'] is not None:
            sample_workspace['bbox_smoother'] = BboxSmoother(self.options['crop_smooth_time'], video_metadata['temporal_data']['sample_rate'])
        else:
            sample_workspace['bbox_smoother'] = None
        if sample_workspace['detected_bbox'] is None:
            sample_workspace['detected_bbox'] = {'x1': 0., 'y1': 0., 'x2': float(self.options['W']), 'y2': float(self.options['H'])}
            sample_workspace['detected_bbox'] = bbox_expand_aspect_ratio(sample_workspace['detected_bbox'], self.tar_aspect_ratio)
        prev_bbox = sample_workspace['final_bbox']

        # Use method to find bounding box
        bbox = None
        if self.options['crop_method'] == 's3fd':
            img8 = img2uint8(img)
            bbox = track_face_s3fd(img8, sample_workspace, self.crop_dataset_workspace, device=device)
        elif 'dlib' in self.options['crop_method']:
            # Initialize tracker.
            if 'tracker' not in self.crop_dataset_workspace:
                if self.options['crop_method'] == 'dlib_hog':
                    self.crop_dataset_workspace['tracker'] = dlib.get_frontal_face_detector()
                elif self.options['crop_method'] == 'dlib_cnn':
                    self.crop_dataset_workspace['tracker'] = dlib.cnn_face_detection_model_v1("face_tracker/mmod_human_face_detector.dat")
                else:
                    raise Exception('Unknown dlib face tracker ' + self.options['crop_method'])
            img8 = img2uint8(img)
            faces = self.crop_dataset_workspace['tracker'](img8, 1)
            if (len(faces) > 0):
                face = faces[0]
                if self.options['crop_method'] == 'dlib_cnn':
                    face = face.rect
                bbox = {'x1': face.left(), 'y1': face.top(), 'x2': face.right(), 'y2': face.bottom()}
        elif self.options['crop_method'] == 'bbox':
            if frame_idx < len(video_metadata['temporal_data']['data']):
                x1 = video_metadata['temporal_data']['data']['face_bbox_x'][frame_idx]
                y1 = video_metadata['temporal_data']['data']['face_bbox_y'][frame_idx]
                bbox_width = video_metadata['temporal_data']['data']['face_bbox_width'][frame_idx]
                bbox_height = video_metadata['temporal_data']['data']['face_bbox_height'][frame_idx]
                bbox = {'x1': x1, 'y1': y1, 'x2': x1 + bbox_width, 'y2': y1 + bbox_height}
        else:
            raise Exception('Unknown crop method ' + self.options['crop_method'])
        
        # If bbox was not detected, use prior detection (initialized to full frame)
        if bbox is not None:
            sample_workspace['detected_bbox'] = bbox_expand_aspect_ratio(bbox, self.tar_aspect_ratio)
            
        # Border
        sample_workspace['border_bbox'] = bbox_add_border(sample_workspace['detected_bbox'], self.options['crop_border'])

        # Deadzone
        if self.options['crop_deadzone'] != 0:
            if sample_workspace['target_bbox'] is None or not bbox_inside_bbox(sample_workspace['border_bbox'], sample_workspace['target_bbox']):
                sample_workspace['target_bbox'] = bbox_add_border(sample_workspace['border_bbox'], [self.options['crop_deadzone'],])
        else:
            sample_workspace['target_bbox'] = sample_workspace['border_bbox']

        # Smoothing
        if sample_workspace['bbox_smoother'] is not None:
            if prev_bbox is None:
                prev_bbox = sample_workspace['target_bbox']
            sample_workspace['final_bbox'] = sample_workspace['bbox_smoother'].smooth(prev_bbox, sample_workspace['target_bbox'])
        else:
            sample_workspace['final_bbox'] = sample_workspace['target_bbox']

        # Generate subset grid
        if sample_workspace['final_bbox'] is not None:
            sample_workspace['subset_grid'] = self.generate_subset_grid(sample_workspace['final_bbox'], original_width, original_height)
        
        # Perform cropping
        if sample_workspace['subset_grid'] is not None:
            out_type = img.dtype
            out_type_info = np.iinfo(out_type)
            img = tr.from_numpy(img.astype(np.float32)).permute(2,0,1)[None, :]
            img = F.grid_sample(img, sample_workspace['subset_grid'], mode='bilinear', padding_mode='zeros', align_corners=False)
            img = img[0, :].permute(1,2,0).numpy()
            img = np.rint(img)
            img = np.clip(img, out_type_info.min, out_type_info.max)
            img = img.astype(out_type)
        else:
            img = cv2.resize(img, (self.options['W'], self.options['H']), interpolation=cv2.INTER_AREA)
        return img

    # Extract the features for one sample
    def extract_sample(self, sample_metadata):
        # Determine which device to use
        use_device = self.options['available_gpus'].pop()
        
        # Setup feature extraction workspace and paths
        crop_sample_workspace = {}
        sample_path = sample_metadata['sample_path']
        video_cropped_base = os.path.join(sample_path, self.options['video_source'] + '_crop_' + self.options['crop_method'])
        video_cropped_path = video_cropped_base + '.avi'
        video_bbox_log_path = video_cropped_base + '_bbox.csv'
        video_metadata = sample_metadata['videos'][self.options['video_source']]
        video_path = video_metadata['path']

        # Only extract if overriding, or some frames or the bbox file are missing
        if self.options['crop_overwrite'] or not os.path.isfile(video_cropped_path) or not os.path.isfile(video_bbox_log_path):
            # Remove and restore extraction directory, if crop_overwrite
            print('Extracting features for {:s}'.format(video_cropped_base))
            
            # Loop through all frames
            try:
                reader = FrameReader(video_path, 0, video_metadata['temporal_data']['total_entries'], video_metadata['format'], target_format='rgb')
                bbox_log = []
                fourcc = cv2.VideoWriter_fourcc(*'FFV1')
                crop_res = (self.options['W'], self.options['H'])
                with VideoWriterContext(video_cropped_path, fourcc, video_metadata['temporal_data']['sample_rate'], crop_res) as video_writer:
                    for frame_idx, img in enumerate(reader):
                        # Crop frame
                        cropped_img = self.crop_frame(img, video_metadata, frame_idx, crop_sample_workspace, device=use_device)
                        video_writer.write(img2uint8(cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)))

                        # Log bbox
                        bbox_log_row = {}
                        for bbox_type in ['detected_bbox', 'border_bbox', 'target_bbox', 'final_bbox']:
                            if crop_sample_workspace[bbox_type] is None:
                                out_bbox = {'x1': None, 'y1': None, 'x2': None, 'y2': None}
                            else:
                                out_bbox = crop_sample_workspace[bbox_type]
                            bbox_type_row = {f"{bbox_type}_{key}": val for key, val in out_bbox.items()}
                            bbox_log_row.update(bbox_type_row)
                        bbox_log.append(bbox_log_row)
            except Exception as e:
                traceback.print_exc()
                print('Failed to extract features for ' + video_path + ': '+ str(e))
            else:
                # Write BBoxes
                if not os.path.isfile(video_bbox_log_path):
                    with open(video_bbox_log_path, 'w')  as output_file:
                        dict_writer = csv.DictWriter(output_file, bbox_log[0].keys())
                        dict_writer.writeheader()
                        dict_writer.writerows(bbox_log)

        # Get subject vector
        subject_encoding_path = os.path.join(sample_path, self.options['video_source'] + '_subject_encoding.csv')
        if not os.path.isfile(subject_encoding_path):
            reader = FrameReader(video_path, 0, video_metadata['temporal_data']['total_entries'], video_metadata['format'], target_format='rgb')
            frame = next(reader)
            boxes = face_recognition.face_locations(frame, model='hog')
            encodings = face_recognition.face_encodings(frame, boxes)
            if len(encodings) < 1:
                raise Exception('Subject encoding not found')
            with open(subject_encoding_path, 'w') as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                writer.writerow(encodings[0])
                print('Created subject encoding file for ' + sample_path)

        # Return the device to the pool
        self.options['available_gpus'].append(use_device)        
