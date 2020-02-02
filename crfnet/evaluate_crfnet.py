#!/usr/bin/env python3
"""
This script runs evaluates a trained neural network model with a given config on the validation data of a data generator.
It outputs precision recall curves with their corresponding AP and weigthed mAP as well as mADE over distance chart.

Arguments:
    --config <PATH TO CONFIG FILE>
    --model <PATH TO MODEL>
    --render to visualize the evaluated samples while running the network
    --read to look for detection pickels in the saved models folder for the corresponding network. 
            If they are not found they are created in order to save computational time in the next run.

"""


### Imports ###
# Standard library imports
import argparse
import os
import sys

# Third party imports

import keras.preprocessing.image
import numpy as np
import matplotlib.pyplot as plt 
import pickle
import math

# Allow relative imports when being executed as script.
if __name__ == "__main__" and not __package__:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    import crfnet  # noqa: F401
    __package__ = "crfnet"

# Local imports
from crfnet.model import architectures
from crfnet.model.architectures.retinanet import retinanet_bbox
from crfnet.utils.config import get_config
from crfnet.utils.keras_version import check_keras_version

from crfnet.utils.eval import _get_annotations, _get_detections, _compute_ap
from crfnet.utils.anchor_calc import compute_overlap
from crfnet.utils.anchor_parameters import AnchorParameters
from crfnet.utils.helpers import makedirs
from crfnet.data_processing.generator.crf_main_generator import create_generators


if __name__ == '__main__':

    FILE_DIRECTORY = os.path.dirname(os.path.abspath(__file__)) 

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=os.path.join(FILE_DIRECTORY,"configs/nuscenes_sample.cfg"))
    parser.add_argument('--model', type=str, default="./saved_models/nuscenes_sample.h5")
    parser.add_argument('--st', type=float, default=None)
    parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('--eval_from_detection_pickle', default=False, action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError("ERROR: Config file \"%s\" not found"%(args.config))
    else:
        cfg = get_config(args.config)

    model_name = args.config.split('/')[-1] 
    model_name = model_name.split('.')[0]
    cfg.model_name = model_name

    if args.st:
        score_threshold = args.st
    else:
        score_threshold = cfg.score_thresh_train

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu

    # create object that stores backbone information
    backbone = architectures.backbone(cfg.network)

    # create the generators
    generators = create_generators(cfg, backbone) 
    test_generator = generators[2]
    
    class_to_color = {
        'bg': np.array([0, 0, 0])/255,
        'human': np.array([34, 114, 227]) / 255,
        'vehicle.bicycle': np.array([0, 182, 0])/255,
        'vehicle.bus': np.array([84, 1, 71])/255,
        'vehicle.car': np.array([189, 101, 0]) / 255,
        'vehicle.motorcycle': np.array([159, 157,156])/255,
        'vehicle.trailer': np.array([0, 173, 162])/255,
        'vehicle.truck': np.array([89, 51, 0])/255
        }

    if args.eval_from_detection_pickle:
        with open ('./saved_models/detection_pickles/' + cfg.model_name, 'rb') as fp:
            data = pickle.load(fp)
            [all_detections, all_annotations] = data
    else:
        # load model
        model = keras.models.load_model(args.model, custom_objects=backbone.custom_objects)
        # load anchor parameters, or pass None (so that defaults will be used)
        if 'small' in cfg.anchor_params:
            anchor_params = AnchorParameters.small
            num_anchors = AnchorParameters.small.num_anchors()
        else:
            anchor_params = None
            num_anchors   = None

        prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params, class_specific_filter=False)
        
        all_detections = _get_detections(test_generator, prediction_model, distance=cfg.distance_detection, 
            score_threshold=score_threshold, max_detections=100, save_path=None, render=args.render,
            distance_scale=100, workers=cfg.workers, cfg=cfg)
        all_annotations = _get_annotations(test_generator)        
        
        pickle_path = './saved_models/detection_pickles'
        makedirs(pickle_path)
        with open(os.path.join(pickle_path, cfg.model_name), 'wb') as fp:
            pickle.dump([all_detections, all_annotations], fp)

    iou_threshold=0.5
    dist_list_all = np.zeros((0,))
    dist_errors_all = np.zeros((0,))

    average_precisions = {}
    recalls = {}
    precisions = {}
    dist_lists = {}
    dist_error_lists = {}
    unoccured_labels = []


    for label in range(test_generator.num_classes()):
        if not test_generator.has_label(label):
            continue

        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        rel_dist_errors = np.zeros((0,))
        dist_list       = np.zeros((0,))
        dist_errors     = np.zeros((0,))
        num_annotations = 0.0

        for i in range(test_generator.size()):
            detections           = all_detections[i][label]
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                score = d[4]
                scores = np.append(scores,score)

                if cfg.distance_detection: 
                    dist = d[5]

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue

                overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations[:,:4])
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                    if cfg.distance_detection:
                        dist_err = np.abs(dist - annotations[assigned_annotation[0],4])
                        rel_dist_err = dist_err / annotations[assigned_annotation[0],4]

                        dist_list = np.append(dist_list, annotations[assigned_annotation[0],4])
                        dist_errors = np.append(dist_errors, dist_err)
                        dist_list_all = np.append(dist_list_all, annotations[assigned_annotation[0],4])
                        dist_errors_all = np.append(dist_errors_all, dist_err)
                        rel_dist_errors = np.append(rel_dist_errors, rel_dist_err)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)

        if num_annotations == 0:
            unoccured_labels.append(label)
            print("No  object of label category {} was detected in the dataset".format(label))
            continue

        # sort by score
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision  = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations
        if len(recall) == 0:
            recall = 0
        if len(precision) == 0:
            precision = 0
        recalls[label] = recall
        precisions[label] = precision
        dist_lists[label] = dist_list
        dist_error_lists[label] = dist_errors
    
    # Calculate mAP
    total_instances = []
    aps = []
    for _, (average_precision, num_annotations ) in average_precisions.items():
        total_instances.append(num_annotations)
        aps.append(average_precision)

    mean_ap = sum([a * b for a, b in zip(total_instances, aps)]) / sum(total_instances)

    ## Store precision recall values
    save_path = './' + cfg.model_name + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data_out = os.path.join(save_path, "pre_rec_category_" + cfg.model_name + ".csv")
    with open(data_out, 'w') as csvFile:
        csvFile.write(",".join(["label", "recall", "precision"])+"\n")

        for label in range(test_generator.num_classes() - 1):
            if label in unoccured_labels:
                continue
            label_name = test_generator.label_to_name(label)
            eval_data =   np.stack(([label_name]*len(recalls[label]), recalls[label], precisions[label]))
            np.savetxt(csvFile, np.swapaxes(eval_data, 0, 1), fmt='%s', delimiter=',')


    # Precision recall curve to file
    col_row = int(np.ceil(math.sqrt(test_generator.num_classes())))
    fig, axs = plt.subplots(col_row, col_row)
    fig.set_figheight(10)
    fig.set_figwidth(15)

    fig.suptitle(cfg.model_name + ' mAP: ' + str(round(mean_ap, 4)))

    if cfg.distance_detection: fig_dist, axs_dist = plt.subplots()

    for label in range(test_generator.num_classes() - 1):
        if label in unoccured_labels:
            continue

        axs[label%col_row, label//col_row].plot(recalls[label], precisions[label])
        axs[label%col_row, label//col_row].set_title(test_generator.label_to_name(label) + ' (AP:' + str(average_precisions[label][0])[:5] + ')', fontsize=10)

        if cfg.distance_detection:
            dist_list = dist_lists[label]
            dist_errors = dist_error_lists[label]
            axs_dist.scatter(dist_list, dist_errors, marker='+', color=class_to_color[test_generator.label_to_name(label)], label=test_generator.label_to_name(label))
            if dist_list.shape[0] > 0:
                z = np.polyfit(dist_list, dist_errors, 1)
                p = np.poly1d(z)
                axs_dist.plot(dist_list, p(dist_list), linestyle='--', linewidth=1, color=class_to_color[test_generator.label_to_name(label)])

    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(save_path + cfg.model_name + ".png"))