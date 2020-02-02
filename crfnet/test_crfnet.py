#!/usr/bin/env python3

"""
This script runs the inference of a trained neural network model with a given config on the samples of a data generator.

Arguments:
    --config <PATH TO CONFIG FILE>
    --model <PATH TO MODEL>
    --st <SCORE THRESHOLD>
    --inference to run the network on all samples (not only the labeled ones)
    --render to visualize the evaluated samples while running the network

Usage:
python crfnet/test_crfnet_retina.py --config <PATH TO CONFIG> --model <PATH TO MODEL>

Exmaple:
python crfnet/test_crfnet_retina.py --config crfnet/configs/inference.cfg --model saved_models/2019-05-03-19-14-28_perfect_filter_nms_cls.h5
"""

### Imports ###
# Standard library imports
import argparse
import os
import sys
import warnings
from datetime import datetime

# Third party imports
import keras
import keras.preprocessing.image
import tensorflow as tf
import progressbar
import cv2
import numpy as np
import pprint
import time
import multiprocessing

# Allow relative imports when being executed as script.
if __name__ == "__main__" and not __package__:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    import crfnet  # noqa: F401
    __package__ = "crfnet"

# Local application imports
from crfnet.model import architectures
from crfnet.model.architectures.retinanet import retinanet_bbox
from crfnet.utils.config import get_config
from crfnet.utils.keras_version import check_keras_version
from crfnet.data_processing.fusion.fusion_projection_lines import create_imagep_visualization
from crfnet.utils.anchor_parameters import AnchorParameters
from crfnet.utils.colors import tum_colors
from crfnet.data_processing.generator.crf_main_generator import create_generators

def visualize_predictions(predictions, image_data_vis, generator, dist=False, verbose=False):
    """
    Visualizes the predictions as bounding boxes with distances or confidence score in a given image.

    :param predictions:         <list>              List with [bboxes, probs, labels]
    :param image_data_vis:      <np.array>          Image where the predictions should be visualized
    :param generator:           <Generator>         Data generator used for name to label mapping
    :dist:                      <bool>              True if distance detection is enabled
    :verbose:                   <bool>              True if detetions should be printed 

    """
    
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 0.4

    # Visualization prediction
    all_dets = []
    [bboxes, probs, labels] = predictions



    for jk in range(bboxes.shape[1]):
        (x1, y1, x2, y2) = bboxes[0,jk,:]
        
        key = generator.label_to_name(labels[0,jk])
        color = class_to_color[key] *255
        cv2.rectangle(image_data_vis,(x1, y1), (x2, y2), color,2)

        if dist is not False:
            textLabel = '{0}: {1:3.1f} {2}'.format(key.split('.', 1)[-1], dist[0,jk], 'm')
            all_dets.append((key,100*probs[0,jk], dist[0,jk]))
        else:
            textLabel = '{}: {}'.format(key.split('.', 1)[-1],int(100*probs[0,jk]))
            all_dets.append((key,100*probs[0,jk]))

        (retval,baseLine) = cv2.getTextSize(textLabel, font, fontScale,1)

        textOrg = int(x1), int(y1)

        cv2.rectangle(image_data_vis, (textOrg[0] - 1,textOrg[1]+baseLine - 1), (textOrg[0]+retval[0] + 1, textOrg[1]-retval[1] - 1), color, -1)
        cv2.putText(image_data_vis, textLabel, textOrg, cv2.FONT_HERSHEY_SIMPLEX, fontScale, (1,1,1), 1)
    
    if verbose: pprint.pprint(all_dets)

    
if __name__ == '__main__':

    FILE_DIRECTORY = os.path.dirname(os.path.abspath(__file__)) 

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=os.path.join(FILE_DIRECTORY,"configs/camra_sample.cfg"))
    parser.add_argument('--model', type=str, default="./saved_models/camra_sample.h5")
    parser.add_argument('--st', type=float, default=None)
    parser.add_argument('--inference', default=False, action='store_true')
    parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('--no_radar_visualization', default=True, action='store_false')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError("ERROR: Config file \"%s\" not found"%(args.config))
    else:
        cfg = get_config(args.config)

    model_name = args.config.split('/')[-1] 
    model_name = model_name.split('.')[0]
    cfg.model_name = cfg.runtime + "_" + model_name
    if args.st:
        score_threshold = args.st
    else:
        score_threshold = cfg.score_thresh_train
    is_radar_visualization = args.no_radar_visualization
    # create object that stores backbone information
    backbone = architectures.backbone(cfg.network)

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu

    # create the generators
    _, _, test_generator, _, _ = create_generators(cfg, backbone)
    generator = test_generator

    model = keras.models.load_model(args.model, custom_objects=backbone.custom_objects)

    # load anchor parameters, or pass None (so that defaults will be used)
    if 'small' in cfg.anchor_params:
        anchor_params = AnchorParameters.small
        num_anchors = AnchorParameters.small.num_anchors()
    else:
        anchor_params = None
        num_anchors   = None

    prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params, class_specific_filter=False)

    use_multiprocessing = cfg.workers > 0 

    if use_multiprocessing:
        enqueuer = keras.utils.data_utils.OrderedEnqueuer(generator, use_multiprocessing=use_multiprocessing, shuffle=False)
        enqueuer.start(workers=cfg.workers, max_queue_size=multiprocessing.cpu_count())
        val_generator = enqueuer.get()

    if generator.num_classes() == len(tum_colors):
        class_to_color = tum_colors
    else:
        # the number of classes if different, 
        # so we need to create the colors dynamically 
        class_to_color = {generator.label_to_name(v): (np.random.rand(3)/3) for v in range(generator.num_classes())}

    save_path = './' + cfg.model_name + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # iterating over all samples in the generator and create the predictions
    for i in progressbar.progressbar(range(generator.size()), prefix='Running network: '):
        if use_multiprocessing:
            inputs, _ = next(val_generator)
        else:
            inputs, _ = generator.compute_input_output([i])
        #inputs, outputs = generator.compute_input_output([i], inference=cfg.inference)

        viz_image = generator.load_image(i)
        if not is_radar_visualization:
            viz_image = viz_image[:,:,:3]
        viz_image = create_imagep_visualization(viz_image, cfg=cfg)

        # run network
        if cfg.distance_detection:
            boxes, scores, labels, dists = prediction_model.predict_on_batch(inputs)
        else:
            boxes, scores, labels = prediction_model.predict_on_batch(inputs)[:3]
        
        selection = np.where(scores > score_threshold)[1]
        boxes = boxes[:,selection,:]
        scores = scores[:,selection]
        labels = labels[:,selection]
        predictions = [boxes, scores, labels]        
        
        if cfg.distance_detection:
            dists = dists[:,selection]
            dists *= 100
            dists = np.squeeze(dists, axis=2)
            visualize_predictions(predictions, viz_image, generator, dist=dists)
        else:
            visualize_predictions(predictions,viz_image, generator)

        if args.render: cv2.imshow('Prediction', viz_image)
        cv2.imwrite(save_path + str(i).zfill(4) + '.png', viz_image)
        cv2.waitKey(1)