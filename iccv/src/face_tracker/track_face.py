import os
import sys
import torch

sys.path.append('s3fd')

from .s3fd.net_s3fd import S3fd_Model
from .s3fd.detect_faces import detect_faces

script_path = os.path.dirname(os.path.realpath(__file__))
model_weight_path = os.path.join(script_path, 's3fd_convert.pth')


def track_face_s3fd(img, sample_workspace, dataset_workspace, device='cuda', face_size=3, confidence=0.75, decay_rate=0.98):
    if 'net' not in dataset_workspace:
        dataset_workspace['net'] = S3fd_Model()
        dataset_workspace['net'].load_state_dict(torch.load(model_weight_path))
        dataset_workspace['net'].to(device)
        dataset_workspace['net'].eval()
    if 'fbox_list' not in sample_workspace:
        sample_workspace['fbox_list'] = []

    bbox_list = []
    with torch.no_grad():
        bbox_list = detect_faces(dataset_workspace['net'], img, face_size, device=device)
    if len(bbox_list) > 0:
        if sample_workspace['detected_bbox'] is None:
            closest_bbox = bbox_list[0]
        else:
            closest_bbox = None
            closest_dist = None
            for candidate_bbox in bbox_list:
                x1_diff = candidate_bbox[0] - sample_workspace['detected_bbox']['x1']
                y1_diff = candidate_bbox[1] - sample_workspace['detected_bbox']['y1']
                x2_diff = candidate_bbox[2] - sample_workspace['detected_bbox']['x2']
                y2_diff = candidate_bbox[3] - sample_workspace['detected_bbox']['y2']
                candidate_dist = (x1_diff * x1_diff) + (y1_diff * y1_diff) + (x2_diff * x2_diff) + (y2_diff * y2_diff)
                if closest_bbox is None or candidate_dist < closest_dist:
                    closest_dist = candidate_dist
                    closest_bbox = candidate_bbox
        return {'x1': closest_bbox[0], 'y1': closest_bbox[1], 'x2': closest_bbox[2], 'y2': closest_bbox[3]}
    else:
        return None
