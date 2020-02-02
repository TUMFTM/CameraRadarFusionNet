"""
This is for preprocessing the radar. e.g. normalization

The std_map and mean_map has been calculated by 
using all radar data from nuScenes
"""

# 3rd Party Libraries
import numpy as np

# Constants
MINIMAL_STD = 0.1

# Local Libraries
channel_map = {
    0: 'x',
    1: 'y',
    2: 'z',
    3: 'dyn_prop',
    4: 'id',
    5: 'rcs',
    6: 'vx',
    7: 'vy',
    8: 'vx_comp',
    9: 'vy_comp',
    10: 'is_quality_valid',
    11: 'ambig_state',
    12: 'x_rms',
    13: 'y_rms',
    14: 'invalid_state',
    15: 'pdh0',
    16: 'vx_rms',
    17: 'vy_rms',
    18: 'distance',
    19: 'azimuth',
    20: 'vrad_comp'
 }

mean_map =  {
    'ambig_state': 3.0,
    'azimuth' : 0.0,
    'distance': 53.248833,
    'dyn_prop': 1.6733711,
    'id': 50.728333,
    'invalid_state': 0.0,
    'is_quality_valid': 1.0,
    'pdh0': 1.0315487,
    'rcs': 8.17769,
    'vrad_comp' : 0.0,
    'vx': 1.3686545,
    'vx_comp': -0.022377603,
    'vx_rms': 16.305613,
    'vy': 0.049794517,
    'vy_comp': -0.009300133,
    'vy_rms': 3.0,
    'x': 49.94819,
    'x_rms': 19.530258,
    'y': 0.0,
    'y_rms': 19.876368,
    'z': 0.0,
 }


std_map = {
    'ambig_state': 2.3841858e-07,
    'azimuth' : 0.41397777,
    'distance': 36.195225,
    'dyn_prop': 1.3584259,
    'id': 35.54741,
    'invalid_state': 0.0,
    'is_quality_valid': 0.0,
    'pdh0': 0.24920322,
    'rcs': 7.5784483,
    'vrad_comp' : 1.9210424,
    'vx': 6.286945,
    'vx_comp': 1.4791956,
    'vx_rms': 0.5934909,
    'vy': 4.8759995,
    'vy_comp': 0.37958348,
    'vy_rms': 2.3841858e-07,
    'x': 36.01416,
    'x_rms': 0.796998,
    'y': 18.806744,
    'y_rms': 1.2855062,
    'z': 0.0,
 } 

normalizing_mask = {
    'ambig_state': True,
    'dyn_prop': True,
    'id': False,
    'invalid_state': True,
    'is_quality_valid': True,
    'pdh0': True,
    'rcs': True,
    'vx': True,
    'vx_comp': True,
    'vx_rms': True,
    'vy': True,
    'vy_comp': True,
    'vy_rms': True,
    'x': True,
    'x_rms': True,
    'y': True,
    'y_rms': True,
    'z': True,
    'distance' : True,
    'azimuth' : True,
    'vrad_comp' : True
 }

# mapping from name to id
channel_map_inverted = {v:k for k,v in channel_map.items()}


def normalize(channel, value, normalization_interval=(-1,1), sigma_factor=1):
    """
    :param channel: the radar channel of the corresponding radar_channel map
    :param value: <float or numpy.array> the value to normalize
    :param sigma_factor: multiples of sigma used for normalizing the value

    :returns: the normalized channel values
    """
    if isinstance(channel, int):
        # convert channel integer into string
        channel = channel_map[channel]

    if normalizing_mask[channel]:
        std = max(std_map[channel], MINIMAL_STD) # we do not want to divide by 0
        normalized_value = (value - mean_map[channel]) / (std*sigma_factor) # standardize to [-1, 1]
        normalized_value = ((normalized_value + 1) / 2) # standardize to [0, 1]
        normalized_value = (normalized_value * (normalization_interval[1] - normalization_interval[0])) + normalization_interval[0] # normalization interval
        return normalized_value
    else:
        # The value is ignored by the normalizing mask
        return value


def denormalize(channel, value, normalization_interval=(-1,1), sigma_factor=1):
    """
    :param channel: the radar channel of the corresponding radar_channel map
    :param value: <float or numpy.array> the value to normalize
    :param sigma_factor: multiples of sigma used for normalizing the value

    :returns: the normalized channel values
    """
    if isinstance(channel, int):
        # convert channel integer into string
        channel = channel_map[channel]

    if normalizing_mask[channel]:
        std = max(std_map[channel],MINIMAL_STD)

        denormalized_value = (value - normalization_interval[0]) / (normalization_interval[1] - normalization_interval[0]) # [0,1]
        denormalized_value = ((denormalized_value * 2) -1) # standardize to [-1, 1]
        denormalized_value = denormalized_value* (std*sigma_factor) + mean_map[channel]
        return denormalized_value
    else:
        # The value is ignored by the normalizing mask
        return value

def filter_radar_byDist(radar_data, distance):
    """
    :param radar_data: axis0 is channels, axis1 is points
    :param distance: [float] -1 for no distance filtering
    """
    if distance > 0:
        no_of_points = radar_data.shape[1]
        deleter = 0
        for point in range(0, no_of_points):
            dist = np.sqrt(radar_data[0,point - deleter]**2 + radar_data[1,point - deleter]**2)
            if dist > distance:
                radar_data = np.delete(radar_data, point - deleter, 1) 
                deleter += 1
    
    return radar_data

def calculate_distances(radar_data):
    """
    :param radar_data: axis0 is channels, axis1 is points
    """
    dist = np.sqrt(radar_data[0,:]**2 + radar_data[1,:]**2)
    return dist


def enrich_radar_data(radar_data):
    """
    This function adds additional data to the given radar data
    
    :param radar_data: The source data which are used to calculate additional metadata
        Semantics: x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0

    :returns enriched_radar_data:
            [0]: x
            [1]: y
            [2]: z
            [3]: dyn_prop
            [4]: id
            [5]: rcs
            [6]: vx
            [7]: vy
            [8]: vx_comp
            [9]: vy_comp
            [10]: is_quality_valid
            [11]: ambig_state
            [12]: x_rms
            [13]: y_rms
            [14]: invalid_state
            [15]: pdh0
            [16]: vx_rms
            [17]: vy_rms
            [18]: distance
            [19]: azimuth
            [20]: vrad_comp
    """
    assert radar_data.shape[0] == 18, "Channel count mismatch."

    # Adding distance
    # Calculate distance
    dist = np.sqrt(radar_data[0,:]**2 + radar_data[1,:]**2)
    dist = np.expand_dims(dist, axis=0)

    # calculate the azimuth values
    azimuth = np.arctan2(radar_data[1,:], radar_data[0,:]) 
    azimuth = np.expand_dims(azimuth, axis=0)

    # Calculate vrad comp
    radial = np.array([radar_data[0,:], radar_data[1,:]]) # Calculate the distance vector
    radial = radial / np.linalg.norm(radial, axis=0, keepdims=True)# Normalize these vectors
    v = np.array([radar_data[8,:], radar_data[9,:]]) # Create the speed vector
    vrad_comp = np.sum(v*radial, axis=0, keepdims=True) # Project the speed component onto this vector

    data_collections = [
        radar_data,
        dist,
        azimuth,
        vrad_comp
    ]

    enriched_radar_data = np.concatenate(data_collections, axis=0)

    return enriched_radar_data
