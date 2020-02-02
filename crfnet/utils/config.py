"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

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
import configparser
from datetime import datetime
import pprint
import subprocess
import json
import os 

# 3rd Party
import keras
import numpy as np

# Local Libraries
from ..utils.anchor_parameters import AnchorParameters
from ..utils.helpers import makedirs

def read_config_file(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)

    return config


def parse_anchor_parameters(config):
    ratios  = np.array(list(map(float, config['anchor_parameters']['ratios'].split(' '))), keras.backend.floatx())
    scales  = np.array(list(map(float, config['anchor_parameters']['scales'].split(' '))), keras.backend.floatx())
    sizes   = list(map(int, config['anchor_parameters']['sizes'].split(' ')))
    strides = list(map(int, config['anchor_parameters']['strides'].split(' ')))

    return AnchorParameters(sizes, strides, ratios, scales)


def get_config(config_file):
    import configparser
    import ast

    config = configparser.ConfigParser()
    config.read(config_file)

    commit = str(subprocess.check_output(["git", "rev-parse", "HEAD"]), 'utf-8').strip()
    branch = str(subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip(), 'utf-8').strip()

    class Configuation():
        def __init__(self):
            self.runtime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            self.git_commit = commit
            self.git_branch = branch
            self.save_model = config['PATH']['save_model']
            self.load_model = config['PATH']['load_model']
            self.tensorboard = config.getboolean('TENSORBOARD', 'tensorboard')
            self.tb_logdir = config['TENSORBOARD']['logdir']
            self.histogram = config.getboolean('TENSORBOARD', 'histogram')
            self.gpu = config['COMPUTING']['gpu']
            self.workers = config.getint('COMPUTING', 'workers')
            self.learning_rate = config.getfloat('HYPERPARAMETERS', 'learning_rate')
            self.batchsize = config.getint('HYPERPARAMETERS', 'batchsize')
            self.epochs = config.getint('HYPERPARAMETERS', 'epochs')
            self.channels = ast.literal_eval(config.get('CRF-Net', 'channels'))
            self.image_size = (config.getint('CRF-Net', 'image_height'), config.getint('CRF-Net', 'image_width'))
            self.dropout_radar = config.getfloat('CRF-Net', 'dropout_radar')
            self.dropout_image = config.getfloat('CRF-Net', 'dropout_image')
            self.network = config['CRF-Net']['network']
            self.pretrain_basenet = config.getboolean('CRF-Net', 'pretrain_basenet')
            self.distance_detection = config.getboolean('CRF-Net', 'distance_detection')
            try:
                self.inference = config.getboolean('CRF-Net', 'inference')
            except:
                self.inference = False
            try:
                self.gpu_mem_usage = config.getfloat('COMPUTING', 'gpu_mem_usage')
            except:
                self.gpu_mem_usage = None
            self.data_set = config['DATA']['data_set']
            self.data_path = config['DATA']['data_path']
            self.n_sweeps = config.getint('DATA','n_sweeps')
            self.weighted_map = config.getboolean('HYPERPARAMETERS', 'weighted_map')
            try:
                self.normalize_radar = config.getboolean('PREPROCESSING', 'normalize_radar')
            except ValueError:
                # we also allow integer
                self.normalize_radar = config.getint('PREPROCESSING', 'normalize_radar')

            self.random_transform = config.getboolean('PREPROCESSING', 'random_transform')
            self.model_name = None

            try:
                self.anchor_box_scales = ast.literal_eval(config.get('FRCNN', 'anchor_box_scales'))
                self.anchor_box_ratios = ast.literal_eval(config.get('FRCNN', 'anchor_box_ratios'))
                self.anchors = ast.literal_eval(config.get('FRCNN', 'anchors'))
                self.num_rois = config.getint('FRCNN', 'num_rois')
                self.rpn_stride = config.getint('FRCNN', 'rpn_stride')
                self.std_scaling = config.getfloat('FRCNN', 'std_scaling')
                self.classifier_regr_std = ast.literal_eval(config.get('FRCNN', 'classifier_regr_std'))
                self.rpn_min_overlap = config.getfloat('FRCNN', 'rpn_min_overlap')
                self.rpn_max_overlap = config.getfloat('FRCNN', 'rpn_max_overlap')
                self.classifier_min_overlap = config.getfloat('FRCNN', 'classifier_min_overlap')
                self.classifier_max_overlap = config.getfloat('FRCNN', 'classifier_max_overlap')
            except:
                pass

            try:
                self.radar_filter_dist = config.getfloat('DATA', 'radar_filter_dist')
            except:
                self.radar_filter_dist = None


            try:
                self.decay = config.getfloat('HYPERPARAMETERS', 'decay')
            except:
                self.decay = 0.0
            try:
                self.kernel_init = config['HYPERPARAMETERS']['kernel_init']
            except:
                self.kernel_init = 'zero'
            try:
                self.category_mapping = dict(config['CATEGORY_MAPPING'])
            except:
                self.category_mapping = {
                    "vehicle.car" : "vehicle.car",
                    "vehicle.motorcycle" : "vehicle.motorcycle",
                    "vehicle.bicycle" : "vehicle.bicycle",
                    "vehicle.bus" : "vehicle.bus",
                    "vehicle.truck" : "vehicle.truck",
                    "vehicle.emergency" : "vehicle.truck",
                    "vehicle.trailer" : "vehicle.trailer",
                    "human" : "human", }
            try:
                self.class_weights = dict(config['CLASS_WEIGHTS'])
            except:
                self.class_weights = None
            try:
                self.distance_alpha = config.getfloat('CRF-Net', 'distance_alpha')
            except:
                self.distance_alpha = 1.0
            try: 
                self.scene_selection = config['DATA']['scene_selection']
            except:
                self.scene_selection = config['DATA']['val_indices']
            try:
                self.sample_selection = config.getboolean('PREPROCESSING', 'sample_selection')
            except:
                self.sample_selection = False
            try:
                self.only_radar_annotated = config.getint('PREPROCESSING', 'only_radar_annotated')
            except:
                self.only_radar_annotated = False
            try:
                self.save_val_img_path = config['DATA']['save_val_img_path']
            except:
                self.save_val_img_path = None
            try:
                self.noise_filter_model = config['DATA']['noise_filter_model']
            except:
                self.noise_filter_model = None
            try:
                self.noise_filter_cfg = config['DATA']['noise_filter_cfg']
            except:
                self.noise_filter_cfg = None
            try:
                self.noise_filter_perfect = config.getboolean('DATA','noise_filter_perfect')
            except:
                self.noise_filter_perfect = False
            try:
                self.noise_filter_threshold = config.getfloat('DATA', 'noise_filter_threshold')
            except:
                self.noise_filter_threshold = 0.0
            try:
                self.class_specific_nms = config.getboolean('CRF-Net', 'class_specific_nms')
            except:
                self.class_specific_nms = False
            try:
                self.score_thresh_train = config.getfloat('CRF-Net', 'score_thresh_train')
            except:
                self.score_thresh_train = 0.05
            try:
                self.noisy_image_method = config['PREPROCESSING']['noisy_image_method']
            except:
                self.noisy_image_method = None
            try:
                self.noise_factor = config.getfloat('PREPROCESSING', 'noise_factor')
            except:
                self.noise_factor = 0.0
            try:
                self.radar_projection_height = config.getfloat('DATA', 'radar_projection_height')
            except:
                self.radar_projection_height = 3
            try:
                self.network_width = config.getfloat('CRF-Net', 'network_width')
            except:
                self.network_width = 1.0
            try:
                self.pooling = config['CRF-Net']['pooling']
            except:
                self.pooling = 'max'
            try:
                self.fusion_blocks = ast.literal_eval(config.get('CRF-Net', 'fusion_blocks'))
            except:
                self.fusion_blocks = [0,1,2,3,4,5]

            if 'coco' in self.data_set:
                self.class_weights = None
                self.distance_detection = False

            try:
                self.anchor_params = config['CRF-Net']['anchor_params']
            except:
                self.anchor_params = 'default'

            try:
                self.seed = config.getint('COMPUTING', 'seed')
            except:
                self.seed = 0

        def get_description(self):
            attributes = [a for a in dir(self) if not a.startswith('__') and not callable(getattr(self,a))]
            values = [getattr(self, attr) for attr in attributes]

            out_string = ""

            for i, attr in enumerate(attributes):
                out_string += "%s = %s\n\n"%(attr, pprint.pformat(values[i]))
            
            return out_string


    cfg = Configuation()
    return cfg

