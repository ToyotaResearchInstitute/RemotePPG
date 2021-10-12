import copy


def bbox_add_border(bbox, crop_border):
    bbox_width = bbox['x2'] - bbox['x1']
    bbox_height = bbox['y2'] - bbox['y1']
    border_width = bbox_width * crop_border[0]
    border_height = bbox_height * crop_border[1 if len(crop_border) > 1 else 0]
    out_bbox = {}
    out_bbox['x1'] = bbox['x1'] - border_width
    out_bbox['x2'] = bbox['x2'] + border_width
    out_bbox['y1'] = bbox['y1'] - border_height
    out_bbox['y2'] = bbox['y2'] + border_height
    return out_bbox


def bbox_inside_bbox(bbox, larger_bbox):
    return bbox['x1'] >= larger_bbox['x1'] and bbox['x2'] <= larger_bbox['x2'] and \
           bbox['y1'] >= larger_bbox['y1'] and bbox['y2'] <= larger_bbox['y2']

def bbox_interpolate(old_bbox, new_bbox, damping):
    out_bbox = {}
    new_weight = 1.0 - damping
    for key in old_bbox.keys():
        out_bbox[key] = (damping * old_bbox[key]) + (new_weight * new_bbox[key])
    return out_bbox

def bbox_expand_aspect_ratio(bbox, tar_aspect_ratio):
    # Expand to aspect ratio
    out_bbox = copy.deepcopy(bbox)
    cur_W = bbox['x2'] - bbox['x1']
    cur_H = bbox['y2'] - bbox['y1']
    cur_aspect_ratio = cur_W / cur_H
    if cur_aspect_ratio > tar_aspect_ratio:
        # Current bbox overly wide, use width to solve for height
        new_H = cur_W / tar_aspect_ratio
        add_H = (new_H - cur_H) / 2.
        out_bbox['y1'] -= add_H
        out_bbox['y2'] += add_H
    else:
        # Current bbox overly tall, use height to solve for width
        new_W = cur_H * tar_aspect_ratio
        add_W = (new_W - cur_W) / 2.
        out_bbox['x1'] -= add_W
        out_bbox['x2'] += add_W
    return out_bbox

class BboxSmoother():
    def __init__(self, smooth_time, sample_rate):
        self.smooth_time = smooth_time
        self.vel = {}
        self.time_delta = 1. / sample_rate

    def smooth(self, from_bbox, to_bbox):
        # Smooth each component
        out_bbox = {}
        for key in from_bbox.keys():
            # Initialize velocity, if not already initialized
            if key not in self.vel:
                self.vel[key] = 0.

            # Smooth - Based on Game Programming Gems 4 Chapter 1.10
            omega = 2. / self.smooth_time
            x = omega * self.time_delta
            exp = 1. / (1. + x + (0.48 * x * x) + (0.235 * x * x * x))
            change = from_bbox[key] - to_bbox[key]
            temp = (self.vel[key] + (omega * change)) * self.time_delta
            self.vel[key] = (self.vel[key] - (omega * temp)) * exp
            out_bbox[key] = to_bbox[key] + (change + temp) * exp
        return out_bbox                    
        