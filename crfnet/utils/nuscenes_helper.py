#!/usr/bin/env python
# coding: utf-8

# Standard libraries
import sys
import os
import os.path as osp

# 3rd party libraries
from PIL import Image
import numpy as np
from pyquaternion import Quaternion

# Local libraries
import nuscenes
from nuscenes.utils.data_classes import RadarPointCloud, Box
from nuscenes.utils.geometry_utils import box_in_image, view_points, BoxVisibility
from . import radar


#todo deprecated, delete when radar_generator.py is updated
def get_nusc_split_samples(nusc, val_indices, validation_ratio=0.2, sample_limit=None):
    """
    :param val_indices: first, mixed or mixed2. Determines the split mode
    :returns: training indices, validation indices
    """

    samples_count = len(nusc.sample)
    split_index = int((1-validation_ratio) * samples_count)

    if val_indices == 'first':
        print("Taking the first {} samples for validation".format(samples_count-split_index))
        sample_indices_train = range(samples_count-split_index,samples_count)
        sample_indices_val = range(0, samples_count-split_index)
    elif val_indices == 'mixed':
        split_1 = int(validation_ratio/2 * samples_count)
        split_2 = int((1-(validation_ratio/2))*samples_count)
        sample_indices_train = range(split_1, split_2)
        sample_indices_val = list(range(0, split_1)) + list(range(split_2, samples_count))
    elif val_indices =='mixed2':
        split_1 = int(validation_ratio * 0.4 * samples_count)
        split_2 = int((1- validation_ratio * 0.25)*samples_count)
        split_3 = int(0.55*samples_count)
        split_4 = int(0.55*samples_count + 0.35*validation_ratio*samples_count)
        sample_indices_train = list(range(split_1, split_3)) + list(range(split_4, split_2))
        sample_indices_val = list(range(0, split_1)) + list(range(split_2, samples_count)) + list(range(split_3, split_4))
    else:
        sample_indices_train = range(0, split_index)
        sample_indices_val = range(split_index, samples_count)
    
    # limit samples
    if sample_limit:
        limit = min(sample_limit, len(sample_indices_train))
        sample_indices_train = sample_indices_train[0:int(limit*(1-validation_ratio))]
        sample_indices_val = sample_indices_val[0:int(limit*validation_ratio)]

    return sample_indices_train, sample_indices_val



def get_sensor_sample_data(nusc, sample, sensor_channel, dtype=np.float32, size=None):
    """
    This function takes the token of a sample and a sensor sensor_channel and returns the according data
    :param sample: the nuscenes sample dict
    :param sensor_channel: the target sensor channel of the given sample to load the data from
    :param dtype: the target numpy type
    :param size: for resizing the image

    Radar Format:
        - Shape: 19 x n
        - Semantics: 
            [0]: x (1)
            [1]: y (2)
            [2]: z (3)
            [3]: dyn_prop (4)
            [4]: id (5)
            [5]: rcs (6)
            [6]: vx (7)
            [7]: vy (8)
            [8]: vx_comp (9)
            [9]: vy_comp (10)
            [10]: is_quality_valid (11)
            [11]: ambig_state (12)
            [12]: x_rms (13)
            [13]: y_rms (14)
            [14]: invalid_state (15)
            [15]: pdh0 (16)
            [16]: vx_rms (17)
            [17]: vy_rms (18)
            [18]: distance (19)

    Image Format:
        - Shape: h x w x 3
        - Channels: RGB
        - size:
            - [int] size to limit image size
            - [tuple[int]] size to limit image size
    """

    # Get filepath
    sd_rec = nusc.get('sample_data', sample['data'][sensor_channel])
    file_name = osp.join(nusc.dataroot, sd_rec['filename'])

    # Check conditions
    if not osp.exists(file_name):
        raise FileNotFoundError(
            "nuscenes data must be located in %s" % file_name)

    # Read the data
    if "RADAR" in sensor_channel:
        pc = RadarPointCloud.from_file(file_name)  # Load radar points
        data = pc.points.astype(dtype)
        data = radar.enrich_radar_data(data) # enrich the radar data an bring them into proper format
    elif "CAM" in sensor_channel:
        i = Image.open(file_name)

        # resize if size is given
        if size is not None:
            try:
                _ = iter(size)
            except TypeError:
                # not iterable
                # limit both dimension to size, but keep aspect ration
                size = (size, size)
                i.thumbnail(size=size)
            else:
                size = size[::-1]  # revert dimensions
                i = i.resize(size=size)

        data = np.array(i, dtype=dtype)

        if np.issubdtype(dtype, np.floating):
            data = data / 255 # floating images usually are on [0,1] interval

    else:
        raise Exception("\"%s\" is not supported" % sensor_channel)

    return data


def calc_mask(nusc, nusc_sample_data, points3d, category_selection, tolerance=0.0, angle_tolerance=0.0, use_points_in_box2=False):
    """
    :param points3d: <np array of channels x samples]>
    :param category_selection: list of categories, which will be masked
    :param tolerance: cartesian tolerance in meters
    :param angle_tolerances: angular tolerance in rad
    """

    # Create Boxes:
    # _, boxes, camera_intrinsic = nusc.get_sample_data(nusc_sample_data['token'], box_vis_level=nuscenes.utils.geometry_utils.BoxVisibility.ANY)
    boxes = nusc.get_boxes(nusc_sample_data['token'])
    cs_record = nusc.get('calibrated_sensor', nusc_sample_data['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', nusc_sample_data['ego_pose_token'])


    mask = np.zeros(points3d.shape[-1])
    for box in boxes:
        if category_selection is None or box.name in category_selection:

            ##### Transform with respect to current sensor #####
            # Move box to ego vehicle coord system
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(pose_record['rotation']).inverse)

            #  Move box to sensor coord system
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)

            ##### Check if points are inside box #####
            if use_points_in_box2:
                cur_mask = nuscenes.utils.geometry_utils.points_in_box2(box, points3d, wlh_tolerance=tolerance, angle_tolerance=angle_tolerance)
            else:
                cur_mask = nuscenes.utils.geometry_utils.points_in_box(box, points3d)
            mask = np.logical_or(mask, cur_mask)
    
    mask = np.clip(mask, a_min=0, a_max=1)

    return mask
