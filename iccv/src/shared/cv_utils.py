import os
import numpy as np
import cv2
from contextlib import contextmanager

from src.shared.s3_utils import S3FileContext


def img2uint8(img):
    # convert to 8 bit if needed
    if img.dtype is np.dtype(np.uint16):
        scale = 65535.  # 16 bit
        img = cv2.convertScaleAbs(img, alpha=(225. / scale))
    return img


def GetVideoMetadata(video_path):
    video_metadata = {}
    if os.path.isfile(video_path):
        cap = cv2.VideoCapture(video_path)
        video_metadata['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_metadata['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_metadata['temporal_data'] = {}
        video_metadata['temporal_data']['total_entries'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_metadata['temporal_data']['sample_rate'] = float(cap.get(cv2.CAP_PROP_FPS))
    else:
        image_paths = next(os.walk(video_path))[2]
        image_paths.sort()
        im = cv2.imread(os.path.join(video_path, image_paths[0]))
        video_metadata['width'] = im.shape[1]
        video_metadata['height'] = im.shape[0]
        video_metadata['temporal_data'] = {}
        video_metadata['temporal_data']['total_entries'] = len(image_paths)
        time_diff = float(os.path.splitext(image_paths[-1])[0]) - float(os.path.splitext(image_paths[0])[0])
        video_metadata['temporal_data']['sample_rate'] = len(image_paths) / time_diff
    return video_metadata

@contextmanager
def VideoCaptureContext(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()

@contextmanager
def VideoWriterContext(*args, **kwargs):
    writer = cv2.VideoWriter(*args, **kwargs)
    try:
        yield writer
    finally:
        writer.release()

# Reads video frames from a file or directory
def FrameReader(path, start_frame, end_frame, video_format="rgb", target_format="rgb"):
    with S3FileContext(path) as s3_path:
        number_frames = end_frame - start_frame
        if os.path.isdir(path):
            image_paths = os.listdir(s3_path)
            image_paths.sort()
            if video_format == 'nir_alternating':
                end_frame = end_frame * 2
            image_paths = image_paths[start_frame:end_frame]
            prev_frame = None
            for image_path in image_paths:
                full_img_path = os.path.join(s3_path, image_path)
                img = cv2.imread(full_img_path, -1)
                if img is None:
                    raise Exception(f'Failed to read image frame {full_img_path}')
                if video_format == 'nir_alternating':
                    if prev_frame is None:
                        # Currently on lit frame
                        prev_frame = img
                    else:
                        # Currently on dark frame
                        sub_frame = np.minimum(prev_frame, img)
                        diff_frame = prev_frame - sub_frame
                        yield np.stack([prev_frame, img, diff_frame], axis=2)
                        prev_frame = None
                elif video_format == 'bayer':
                    if target_format == 'yuv':
                        yield cv2.cvtColor(img, cv2.COLOR_BayerBG2YUV)
                    else:
                        yield cv2.cvtColor(img, cv2.COLOR_BayerBG2RGB)
                else: # rgb
                    if target_format == 'yuv':
                        yield cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                    else:
                        yield cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            with VideoCaptureContext(s3_path) as cap:
                frame_idx = start_frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                for i in range(number_frames):
                    ret, img = cap.read()
                    if not ret:
                        raise Exception(f'Failed to read video frame {frame_idx} in {s3_path}')
                    frame_idx += 1
                    if target_format == 'yuv':
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                    else:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    yield img
