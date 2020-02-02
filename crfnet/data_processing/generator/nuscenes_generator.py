"""
Copyright 2017-2018 Fellfalla (https://github.com/Fellfalla/)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Original Source: https://github.com/fizyr/keras-retinanet
"""

# Standard Libraries
import csv
import os
import sys
import math

# 3rd Party Libraries
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
import progressbar

# Local Libraries
# Allow relative imports when being executed as script.
if __name__ == "__main__" and not __package__:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    import crfnet.data_processing.generator  # noqa: F401
    __package__ = "crfnet.data_processing.generator"

# Local imports
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud, Box
from nuscenes.utils.geometry_utils import box_in_image, view_points, BoxVisibility, points_in_box

from .generator import Generator
from ...utils import radar
from ...utils.nuscenes_helper import get_sensor_sample_data, calc_mask
from ...data_processing.fusion.fusion_projection_lines import create_imagep_visualization
from ...utils.noise_img import noisy

class NuscenesGenerator(Generator):
    """ Generate data for a nuScenes dataset.

    See www.nuscenes.org for more information.
    """
    DATATYPE = np.float32

    def __init__(
        self,
        nusc,
        scene_indices=None,
        channels=[0,1,2],
        category_mapping=None,
        radar_input_name=None,
        radar_width=None,
        image_radar_fusion=True,
        camera_dropout=0.0,
        radar_dropout=0.0,
        normalize_radar=False,
        sample_selection=False,
        only_radar_annotated=False,
        n_sweeps=1, 
        noise_filter=None,
        noise_filter_threshold=0.5,
        noisy_image_method=None,
        noise_factor=0,
        perfect_noise_filter=False,
        noise_category_selection=None,
        inference=False,
        **kwargs
    ):
        """ Initialize a nuScenes data generator.
        : use param config for giving the arguments to this class

        :param nusc: Object pointing at a nuscenes database
        :param sample_indices: <int> Which samples to take from nusc database
        :param channels: Which image and radar channels to use. Only in combination with
            image_radar_fusion=True will return image_plus data. Otherwise the given channels
            are split into radar_infeed and image_infeed.
        :param category_mapping: <dict> dictionary between original classes and target classes.
            Only classes given by this dict will be used for annotations. None for all categories.
        :param radar_input_name: <str> name of the input_tensor for radar infeed into the nn
        :param radar_width: width of the radar-data-array
        :param image_radar_fusion: <bool> Determines if the data_generator performs the 
            default image_plus fusion.
        """

        # Parameters
        self.nusc = nusc
        self.dropout_chance = 0.0
        self.radar_sensors = ['RADAR_FRONT']
        self.camera_sensors = ['CAM_FRONT']
        self.labels = {}
        self.image_data = dict()
        self.classes, self.labels = self._get_class_label_mapping([c['name'] for c in nusc.category], category_mapping)
        self.channels = channels
        self.radar_channels = [ch for ch in channels if ch >= 3]
        self.image_channels = [ch for ch in channels if ch < 3]
        self.normalize_bbox = False # True for normalizing the bbox to [0,1]
        self.radar_input_name = radar_input_name
        self.radar_width = radar_width
        self.radar_dropout = radar_dropout
        self.camera_dropout = camera_dropout
        self.sample_selection = sample_selection
        self.only_radar_annotated = only_radar_annotated
        self.n_sweeps = n_sweeps
        self.noisy_image_method = noisy_image_method
        self.noise_factor = noise_factor
        self.cartesian_uncertainty = (0, 0, 0) # meters
        self.angular_uncertainty = math.radians(0) # degree
        self.inference = inference

        #todo we cannot initialize the parent class first, because it depends on size()
        self.image_min_side = kwargs['image_min_side']
        self.image_max_side = kwargs['image_max_side']

        # assign functions
        self.image_radar_fusion = image_radar_fusion
        self.normalize_radar = normalize_radar

        # Optional imports
        self.radar_array_creation = None
        if self._is_image_plus_enabled() or self.camera_dropout > 0.0:
            # Installing vizdom is required
            from crfnet.data_processing.fusion.fusion_projection_lines import imageplus_creation, create_spatial_point_array

            self.image_plus_creation = imageplus_creation
            self.radar_array_creation = create_spatial_point_array

        self.noise_filter_threshold = noise_filter_threshold
        self.perfect_noise_filter = perfect_noise_filter
        self.noise_category_selection = noise_category_selection
        

        # TEST: Create immediately
        if noise_filter and not isinstance(noise_filter, NfDockerClient):
            raise NotImplementedError('Neural Filter not in opensource repository ')
        else:
            self.noise_filter = None

        # Create all sample tokens
        self.sample_tokens = {}
        prog = 0
        progbar = progressbar.ProgressBar(prefix='Initializing data generator: ')
        skip_count = 0



        # Resolve sample indexing
        if scene_indices is None:
            # We are using all scenes
            scene_indices = range(len(nusc.scene))

        assert hasattr(scene_indices, '__iter__'), "Iterable object containing sample indices expected"

        for scene_index in scene_indices:
            first_sample_token = nusc.scene[scene_index]['first_sample_token']
            nbr_samples = nusc.scene[scene_index]['nbr_samples']

            curr_sample = nusc.get('sample', first_sample_token)
            
            for _ in range(nbr_samples):
                self.sample_tokens[prog] = curr_sample['token']
                if curr_sample['next']:
                    next_token = curr_sample['next']
                    curr_sample = nusc.get('sample', next_token)
                prog += 1
                progbar.update(prog)


        if self.sample_selection: print("\nSkipped {} samples due to zero annotations".format(skip_count))
        # Create all annotations and put into image_data
        self.image_data = {image_index:None for image_index in self.sample_tokens}

        # Finalize
        super(NuscenesGenerator, self).__init__(**kwargs)

    @staticmethod
    def _get_class_label_mapping(category_names, category_mapping):
        """
        :param category_mapping: [dict] Map from original name to target name. Subsets of names are supported. 
            e.g. {'pedestrian' : 'pedestrian'} will map all pedestrian types to the same label

        :returns: 
            [0]: [dict of (str, int)] mapping from category name to the corresponding index-number
            [1]: [dict of (int, str)] mapping from index number to category name
        """
        # Initialize local variables
        original_name_to_label = {}
        original_category_names = category_names.copy()
        original_category_names.append('bg')
        if category_mapping is None:
            # Create identity mapping and ignore no class
            category_mapping = dict()
            for cat_name in category_names:
                category_mapping[cat_name] = cat_name

        # List of unique class_names
        selected_category_names = set(category_mapping.values()) # unordered
        selected_category_names = list(selected_category_names)
        selected_category_names.sort() # ordered
      
        # Create the label to class_name mapping
        label_to_name = { label:name for label, name in enumerate(selected_category_names)}
        label_to_name[len(label_to_name)] = 'bg' # Add the background class

        # Create original class name to label mapping
        for label, label_name in label_to_name.items():

            # Looking for all the original names that are adressed by label name
            targets = [original_name for original_name in original_category_names if label_name in original_name]

            # Assigning the same label for all adressed targets
            for target in targets:
                
                # Check for ambiguity
                assert target not in original_name_to_label.keys(), 'ambigous mapping found for (%s->%s)'%(target, label_name)
                
                # Assign label to original name
                # Some label_names will have the same label, which is totally fine
                original_name_to_label[target] = label

        # Check for correctness
        actual_labels = original_name_to_label.values()
        expected_labels = range(0, max(actual_labels)+1) # we want to start labels at 0
        assert all([label in actual_labels for label in expected_labels]), 'Expected labels do not match actual labels'

        return original_name_to_label, label_to_name

    def _is_image_plus_enabled(self):
        """
        True if image radar fusion is enabled and 
        radar channels are requested.
        """
        r = 0 in self.channels
        g = 1 in self.channels
        b = 2 in self.channels
        return self.image_radar_fusion and len(self.channels) > r+g+b

    def size(self):
        """ Size of the dataset.
        """
        return len(self.sample_tokens)

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        return len(self.labels)

    def has_label(self, label):
        """ Return True if label is a known label.
        """
        return label in self.labels

    def has_name(self, name):
        """ Returns True if name is a known class.
        """
        return name in self.classes

    def name_to_label(self, name):
        """ Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.labels[label]

    def inv_label_to_name(self, name):
        """ Map name to label.
        """
        class_dict = {y:x for x,y in self.labels.items()}
        return class_dict[name]
        
    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        All images of nuscenes dataset have the same aspect ratio which is 16/9
        """
        # All images of nuscenes dataset have the same aspect ratio
        return 16/9 
        # sample_token = self.sample_tokens[image_index]
        # sample = self.nusc.get('sample', sample_token)

        # image_sample = self.load_sample_data(sample, camera_name)
        # return float(image_sample.shape[1]) / float(image_sample.shape[0])


    def load_radar_array(self, sample_index, target_width):
        # Initialize local variables
        if not self.radar_array_creation:
            from ..raw_data_fusion.fusion_projection_lines import create_spatial_point_array
            self.radar_array_creation = create_spatial_point_array

        radar_name = self.radar_sensors[0]
        camera_name = self.camera_sensors[0]

        # Gettign data from nuscenes database
        sample_token = self.sample_tokens[sample_index]
        sample = self.nusc.get('sample', sample_token)

        # Grab the front camera and the radar sensor.
        radar_token = sample['data'][radar_name]
        camera_token = sample['data'][camera_name]
        image_target_shape = (self.image_min_side, self.image_max_side)

        # Create the array
        radar_sample = self.load_sample_data(sample, radar_name) # Load samples from disk
        radar_array = self.radar_array_creation(self.nusc, radar_sample, radar_token, camera_token, target_width=target_width)

        return radar_array

    def set_noise_factor(self, noise_factor):
        """
        This function turns off the noise factor: It is useful for rendering. 
        """
        self.noise_factor = noise_factor

    def load_image(self, image_index):
        """
        Returns the image plus from given image and radar samples.
        It takes the requested channels into account.

        :param sample_token: [str] the token pointing to a certain sample

        :returns: imageplus
        """
        # Initialize local variables
        radar_name = self.radar_sensors[0]
        camera_name = self.camera_sensors[0]

        # Gettign data from nuscenes database
        sample_token = self.sample_tokens[image_index]
        sample = self.nusc.get('sample', sample_token)

        # Grab the front camera and the radar sensor.
        radar_token = sample['data'][radar_name]
        camera_token = sample['data'][camera_name]
        image_target_shape = (self.image_min_side, self.image_max_side)

        # Load the image
        image_sample = self.load_sample_data(sample, camera_name)

        # Add noise to the image if enabled
        if self.noisy_image_method is not None and self.noise_factor>0:
            image_sample = noisy(self.noisy_image_method, image_sample, self.noise_factor)

        if self._is_image_plus_enabled() or self.camera_dropout > 0.0:

            # Parameters
            kwargs = {
            'pointsensor_token': radar_token,
            'camera_token': camera_token,
            'height': (0, self.radar_projection_height), 
            'image_target_shape': image_target_shape,
            'clear_radar': np.random.rand() < self.radar_dropout,
            'clear_image': np.random.rand() < self.camera_dropout,
            }
    
            # Create image plus
            # radar_sample = self.load_sample_data(sample, radar_name) # Load samples from disk
        

            # Get filepath
            if self.noise_filter:
                required_sweep_count = self.n_sweeps + self.noise_filter.num_sweeps_required -1
            else:
                required_sweep_count = self.n_sweeps

            # sd_rec = self.nusc.get('sample_data', sample['data'][sensor_channel])
            sensor_channel = radar_name
            pcs, times = RadarPointCloud.from_file_multisweep(self.nusc, sample, sensor_channel, \
                sensor_channel, nsweeps=required_sweep_count, min_distance=0.0, merge=False)
            
            
            if self.noise_filter:
                # fill up with zero sweeps
                for _ in range(required_sweep_count - len(pcs)):
                    pcs.insert(0, RadarPointCloud(np.zeros(shape=(RadarPointCloud.nbr_dims(), 0))))

            radar_sample = [radar.enrich_radar_data(pc.points) for pc in pcs]
            
            if self.noise_filter:
                ##### Filter the pcs #####
                radar_sample = list(self.noise_filter.denoise(radar_sample, self.n_sweeps))

            if len(radar_sample) == 0:
                radar_sample = np.zeros(shape=(len(radar.channel_map),0))
            else:
                ##### merge pcs into single radar samples array #####
                radar_sample = np.concatenate(radar_sample, axis=-1)

            radar_sample = radar_sample.astype(dtype=np.float32)
            
            if self.perfect_noise_filter:
                cartesian_uncertainty = 0.5 # meters
                angular_uncertainty = math.radians(1.7) # degree
                category_selection = self.noise_category_selection

                nusc_sample_data = self.nusc.get('sample_data', radar_token)
                radar_gt_mask = calc_mask(nusc=self.nusc, nusc_sample_data=nusc_sample_data, points3d=radar_sample[0:3,:], \
                    tolerance=cartesian_uncertainty, angle_tolerance=angular_uncertainty, \
                    category_selection=category_selection)

                # radar_sample = radar_sample[:, radar_gt_mask.astype(np.bool)]
                radar_sample = np.compress(radar_gt_mask, radar_sample, axis=-1)


            if self.normalize_radar:
                # we need to noramlize
                # : use preprocess method analog to image preprocessing
                sigma_factor = int(self.normalize_radar)
                for ch in range(3, radar_sample.shape[0]): # neural fusion requires x y and z to be not normalized
                    norm_interval = (-127.5,127.5) # caffee mode is default and has these norm interval for img
                    radar_sample[ch,:] = radar.normalize(ch, radar_sample[ch,:], normalization_interval=norm_interval, sigma_factor=sigma_factor)
                    
            
            img_p_full = self.image_plus_creation(self.nusc, image_data=image_sample, radar_data=radar_sample, **kwargs)
            
            # reduce to requested channels
            #self.channels = [ch - 1 for ch in self.channels] # Shift channels by 1, cause we have a weird convetion starting at 1
            input_data = img_p_full[:,:,self.channels]

        else: # We are not in image_plus mode
            # Only resize, because in the other case this is contained in image_plus_creation
            input_data = cv2.resize(image_sample, image_target_shape[::-1])

        return input_data


    def load_sample_data(self, sample, sensor_channel):
        """
        This function takes the token of a sample and a sensor sensor_channel and returns the according data

        Radar format: <np.array>
            - Shape: 18 x n
            - Semantics: x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0

        Image format: <np.array>
            - Shape: h x w x 3
            - Values: [0,255]
            - Channels: RGB
        """
        return get_sensor_sample_data(self.nusc, sample, sensor_channel, dtype=np.float32, size=None)


    def create_annotations(self, sample_token, sensor_channels):
        """
        Create annotations for the the given sample token.

        1 bounding box vector contains:


        :param sample_token: the sample_token to get the annotation for
        :param sensor_channels: list of channels for cropping the labels, e.g. ['CAM_FRONT', 'RADAR_FRONT']
            This works only for CAMERA atm

        :returns: 
            annotations dictionary:
            {
                'labels': [] # <list of n int>  
                'bboxes': [] # <list of n x 4 float> [xmin, ymin, xmax, ymax]
                'distances': [] # <list of n float>  Center of box given as x, y, z.
                'visibilities': [] # <list of n float>  Visibility of annotated object
            }
        """

        if any([s for s in sensor_channels if 'RADAR' in s]):
            print("[WARNING] Cropping to RADAR is not supported atm")
            sensor_channels = [c for c in sensor_channels if 'CAM' in sensor_channels]

        sample = self.nusc.get('sample', sample_token)
        annotations_count = 0
        annotations = {
            'labels': [], # <list of n int>  
            'bboxes': [], # <list of n x 4 float> [xmin, ymin, xmax, ymax]
            'distances': [], # <list of n float>  Center of box given as x, y, z.
            'visibilities': [],
            'num_radar_pts':[] #<list of n int>  number of radar points that cover that annotation
            }

        # Camera parameters
        for selected_sensor_channel in sensor_channels:
            sd_rec = self.nusc.get('sample_data', sample['data'][selected_sensor_channel])

            # Create Boxes:
            _, boxes, camera_intrinsic = self.nusc.get_sample_data(sd_rec['token'], box_vis_level=BoxVisibility.ANY)
            imsize_src = (sd_rec['width'], sd_rec['height']) # nuscenes has (width, height) convention
            
            bbox_resize = [ 1. / sd_rec['height'], 1. / sd_rec['width'] ]
            if not self.normalize_bbox:
                bbox_resize[0] *= float(self.image_min_side)
                bbox_resize[1] *= float(self.image_max_side)

            # Create labels for all boxes that are visible
            for box in boxes:

                # Add labels to boxes 
                if box.name in self.classes:
                    box.label = self.classes[box.name]
                    # Check if box is visible and transform box to 1D vector
                    if box_in_image(box=box, intrinsic=camera_intrinsic, imsize=imsize_src, vis_level=BoxVisibility.ANY):
                        
                        ## Points in box method for annotation filterS
                        # check if bounding box has an according radar point
                        if self.only_radar_annotated == 2:

                            pcs, times = RadarPointCloud.from_file_multisweep(self.nusc, sample, self.radar_sensors[0], \
                                selected_sensor_channel, nsweeps=self.n_sweeps, min_distance=0.0, merge=False)

                            for pc in pcs:
                                pc.points = radar.enrich_radar_data(pc.points)    

                            if len(pcs) > 0:
                                radar_sample = np.concatenate([pc.points for pc in pcs], axis=-1)
                            else:
                                print("[WARNING] only_radar_annotated=2 and sweeps=0 removes all annotations")
                                radar_sample = np.zeros(shape=(len(radar.channel_map), 0))
                            radar_sample = radar_sample.astype(dtype=np.float32)

                            mask = points_in_box(box, radar_sample[0:3,:])
                            if True not in mask:
                                continue 


                        # If visible, we create the corresponding label
                        box2d = box.box2d(camera_intrinsic) # returns [xmin, ymin, xmax, ymax]
                        box2d[0] *= bbox_resize[1]
                        box2d[1] *= bbox_resize[0]
                        box2d[2] *= bbox_resize[1]
                        box2d[3] *= bbox_resize[0]

                        annotations['bboxes'].insert(annotations_count, box2d)
                        annotations['labels'].insert(annotations_count, box.label)
                        annotations['num_radar_pts'].insert(annotations_count, self.nusc.get('sample_annotation', box.token)['num_radar_pts'])

                        distance =  (box.center[0]**2 + box.center[1]**2 + box.center[2]**2)**0.5
                        annotations['distances'].insert(annotations_count, distance)
                        annotations['visibilities'].insert(annotations_count, int(self.nusc.get('sample_annotation', box.token)['visibility_token']))
                        annotations_count += 1
                else:
                    # The current name has been ignored
                    pass

        annotations['labels'] = np.array(annotations['labels'])
        annotations['bboxes'] = np.array(annotations['bboxes'])
        annotations['distances'] = np.array(annotations['distances'])
        annotations['num_radar_pts'] = np.array(annotations['num_radar_pts'])
        annotations['visibilities'] = np.array(annotations['visibilities'])

        # num_radar_pts mathod for annotation filter
        if self.only_radar_annotated == 1:

            anns_to_keep = np.where(annotations['num_radar_pts'])[0]

            for key in annotations:
                annotations[key] = annotations[key][anns_to_keep]

        return annotations

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        annotations = self.image_data[image_index]

        if annotations is None:
            sample_token = self.sample_tokens[image_index]
            annotations = self.create_annotations(sample_token, self.camera_sensors)

            self.image_data[image_index] = annotations

        return annotations

    def compute_input_output(self, group, inference=False):
        """
        Extends the basic function with the capability to 
        add radar input data to the input batch.
        """
        inputs, targets = super(NuscenesGenerator, self).compute_input_output(group)

        if self.radar_input_name:
            # Load radar data
            radar_input_batch = []
            for sample_index in group:
                radar_array = self.load_radar_array(sample_index, target_width=self.radar_width)
                radar_input_batch.append(radar_array)

            radar_input_batch = np.array(radar_input_batch)

            inputs = {
                'input_1': inputs,
                self.radar_input_name: radar_input_batch
            }

        return inputs, targets



if __name__ == "__main__":
    import cv2
    import argparse
    # Allow relative imports

    from ...utils.anchor_calc import anchor_targets_bbox 
    from ...utils.anchor import guess_shapes, anchors_for_shape, compute_gt_annotations
    from ...utils.anchor_parameters import AnchorParameters
    from ...utils.image import preprocess_image, preprocess_image_inverted
    from ...utils.config import get_config
    from ...utils.visualization import draw_boxes
    from ...utils.transform import random_transform_generator
    from ...model import architectures

    FILE_DIRECTORY = os.path.dirname(os.path.abspath(__file__)) 

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=os.path.join(FILE_DIRECTORY, "../../configs/local.cfg"))
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--sample', type=int, default=0)
    parser.add_argument('--bboxes', dest='bboxes', action='store_true')
    parser.add_argument('--no-bboxes', dest='bboxes', action='store_false')
    parser.set_defaults(bboxes=True)
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError("ERROR: Config file \"%s\" not found"%(args.config))
    else:
        cfg = get_config(args.config)
 
    if cfg.anchor_params:
        if 'small' in cfg.anchor_params:
            anchor_params = AnchorParameters.small
        else:
            anchor_params = None
    else:
        anchor_params = None

    backbone = architectures.backbone(cfg.network)
    common_args = {
        'batch_size'                : cfg.batchsize,
        'config'                    : None,
        'image_min_side'            : cfg.image_size[0],
        'image_max_side'            : cfg.image_size[1],
        'filter_annotations_enabled': False,
        'preprocess_image'          : backbone.preprocess_image,
        'normalize_radar'           : cfg.normalize_radar,
        'camera_dropout'            : cfg.dropout_image,
        'radar_dropout'             : cfg.dropout_radar,
        'channels'                  : cfg.channels,
        'distance'                  : cfg.distance_detection,
        'sample_selection'          : cfg.sample_selection,
        'only_radar_annotated'      : cfg.only_radar_annotated,
        'n_sweeps'                  : cfg.n_sweeps,
        'noise_filter'              : cfg.noise_filter_cfg,
        'noise_filter_threshold'    : cfg.noise_filter_threshold,
        'noisy_image_method'        : cfg.noisy_image_method,
        'noise_factor'              : cfg.noise_factor,
        'perfect_noise_filter'      : cfg.noise_filter_perfect,
        'radar_projection_height'   : cfg.radar_projection_height,
        'noise_category_selection'  : None if cfg.class_weights is None else cfg.class_weights.keys(),
        'inference'                 : cfg.inference,
        'anchor_params'             : anchor_params,
    }

    class_to_color = {
        'bg': np.array([0, 0, 0])/255,
        'human.pedestrian.adult': np.array([34, 114, 227]) / 255,
        'vehicle.bicycle': np.array([0, 182, 0])/255,
        'vehicle.bus': np.array([84, 1, 71])/255,
        'vehicle.car': np.array([189, 101, 0]) / 255,
        'vehicle.motorcycle': np.array([159, 157,156])/255,
        'vehicle.trailer': np.array([0, 173, 162])/255,
        'vehicle.truck': np.array([89, 51, 0])/255,
        }
    
    if 'mini' in cfg.data_set:
        nusc = NuScenes(version='v1.0-mini', dataroot=cfg.data_path, verbose=True)
    else:
        try:
            nusc = NuScenes(version='v1.0-trainval', dataroot=cfg.data_path, verbose=True)
        except:
            nusc = NuScenes(version='v1.0-mini', dataroot=cfg.data_path, verbose=True)

    ## Data Augmentation
    transform_generator = random_transform_generator(
        min_rotation=-0.1,
        max_rotation=0.1,
        min_translation=(-0.1, -0.1),
        max_translation=(0.1, 0.1),
        min_shear=-0.1,
        max_shear=0.1,
        min_scaling=(0.9, 0.9),
        max_scaling=(1.1, 1.1),
        flip_x_chance=0.5,
        flip_y_chance=0.0,
    )

    data_generator = NuscenesGenerator(nusc, 
                                        scene_indices=None, 
                                        category_mapping=cfg.category_mapping, 
                                        transform_generator=transform_generator,
                                        shuffle_groups=False, 
                                        compute_anchor_targets=anchor_targets_bbox,
                                        compute_shapes=guess_shapes,
                                        **common_args)
    
    if data_generator.noise_filter:
        data_generator.noise_filter.render = True

    i = args.sample
    while i < len(data_generator):
        print("Sample ", i)

        # Get the data
        inputs, targets = data_generator[i]
        img = inputs[0]
        img = preprocess_image_inverted(img)
        ann = data_generator.load_annotations(i)

        assert img.shape[0] == common_args['image_min_side']
        assert img.shape[1] == common_args['image_max_side']
        # assert img.shape[2] == len(common_args['channels'])

        # Turn data into vizualizable format
        viz = create_imagep_visualization(img, draw_circles=False, cfg=cfg, radar_lines_opacity=0.9)

        font                   = cv2.FONT_HERSHEY_SIMPLEX
        fontScale              = 0.4
        lineType               = 1

        if args.debug:
            ## Positive Anchor Visualization
            anchors = anchors_for_shape(viz.shape, anchor_params=anchor_params)
            positive_indices, _, max_indices = compute_gt_annotations(anchors, ann['bboxes'])
            draw_boxes(viz, anchors[positive_indices], (255, 255, 0), thickness=1)

            ## Data Augmentation
            viz, ann = data_generator.random_transform_group_entry(viz, ann)
        
        if args.bboxes:
            for a in range(len(ann['bboxes'])):
                label_name = data_generator.label_to_name(ann['labels'][a])
                dist = ann['distances'][a]

                if label_name in class_to_color:
                    color = class_to_color[label_name] * 255
                else:
                    color = class_to_color['bg']

                p1 = (int(ann['bboxes'][a][0]), int(ann['bboxes'][a][1])) # Top left
                p2 = (int(ann['bboxes'][a][2]), int(ann['bboxes'][a][3])) # Bottom right
                cv2.rectangle(viz,p1, p2, color,1)

                textLabel = '{0}: {1:3.1f} {2}'.format(label_name.split('.', 1)[-1], dist, 'm')

                (retval,baseLine) = cv2.getTextSize(textLabel, font, fontScale,1)

                textOrg = p1

                cv2.rectangle(viz, (textOrg[0] - 1,textOrg[1]+baseLine - 1), (textOrg[0]+retval[0] + 1, textOrg[1]-retval[1] - 1), color, -1)
                cv2.putText(viz, textLabel, textOrg, cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255,255,255), 1)


        # Visualize data
        cv2.imshow("Nuscenes Data Visualization",viz)
        # cv2.imwrite('./ground_truth_selected/' + str(i).zfill(4) +'.png', viz*255)
        key = cv2.waitKey(0)
        if key == ord('p'): #previous image
            i = i-1
        elif key == ord('s'):
            print("saving image")
            cv2.imwrite("saved_img.png", viz)
        elif key == ord('n'):
            print("%c -> jump to next scene"%key)
            i = i+40
        elif key == ord('m'):
            print("%c -> jump to previous scene"%key)
            i = i-40
        elif key == ord('q'):
            break
        else:
            i = i+1

        i = max(i, 0)



