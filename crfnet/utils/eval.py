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
"""

from .anchor_calc import compute_overlap
from .visualization import draw_detections, draw_annotations
from crfnet.data_processing.fusion.fusion_projection_lines import create_imagep_visualization
import keras
import numpy as np
import os
import time
from threading import Thread
import queue
import multiprocessing
import sys

import cv2
import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."


data_queue = queue.Queue()

def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_detections(generator, model, distance=False, score_threshold=0.05, max_detections=100, 
    save_path=None, render=False, distance_scale=100, workers=0, cfg=None):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(generator.num_classes()) if generator.has_label(i)] for j in range(generator.size())]


    use_multiprocessing = workers > 0 
    if use_multiprocessing:
        enqueuer = keras.utils.data_utils.OrderedEnqueuer(generator, use_multiprocessing=use_multiprocessing, shuffle=False)
        enqueuer.start(workers=workers, max_queue_size=multiprocessing.cpu_count())
        val_generator = enqueuer.get()
    

    for i in progressbar.progressbar(range(generator.size()), prefix='Running network on {} workers: '.format(workers)):
        if use_multiprocessing:
            inputs, _ = next(val_generator)
        else:
            inputs, _ = generator.compute_input_output([i])

        if save_path or render:
            raw_image = generator.load_image(i)

        # run network
        if distance:
            boxes, scores, labels, dists = model.predict_on_batch(inputs)
            dists = np.squeeze(dists, axis=2)
            dists = dists * distance_scale
        else:
            boxes, scores, labels = model.predict_on_batch(inputs)

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]

        # select those scores
        scores = scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        image_boxes      = boxes[0, indices[scores_sort], :]
        image_scores     = scores[scores_sort]
        image_labels     = labels[0, indices[scores_sort]]
        if distance: 
            image_distances  = dists[0, scores_sort]
            image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_distances, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)
        else:
            image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        # Create Visualization
        if save_path or render:
            viz_image = create_imagep_visualization(raw_image, cfg=cfg)
            #draw_annotations(viz_image, generator.load_annotations(i), label_to_name=generator.label_to_name) # Draw annotations
            draw_detections(viz_image, image_boxes, image_scores, image_labels,score_threshold=0.3, label_to_name=generator.label_to_name) # Draw detections
        
        if render:
            # Show 
            try:
                cv2.imshow("debug", viz_image)
                cv2.waitKey(1)
            except Exception as e:
                print("Render error:")
                print(e)
        if save_path is not None:
            # Store
            cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), viz_image)

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]
    
    if use_multiprocessing:
        enqueuer.stop()
    
    return all_detections


def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]

    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in progressbar.progressbar(range(generator.size()), prefix='Parsing annotations: '):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            if len(annotations['bboxes']) ==0:
                all_annotations[i][label] = annotations['bboxes'][annotations['labels'] == label].copy()
                continue 

            box_and_dist_and_vis = np.concatenate((annotations['bboxes'], np.expand_dims(annotations['distances'], axis=1), \
                np.expand_dims(annotations['visibilities'], axis=1).astype(np.float64)), axis=1)
            all_annotations[i][label] = box_and_dist_and_vis[annotations['labels'] == label].copy()


    return all_annotations


def evaluate(
    generator,
    model,
    distance,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=100,
    save_path=None,
    render=False,
    workers = 0
):
    """ Evaluate a given dataset using a given model.

    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """
    assert generator.shuffle_groups == False, 'validation data must not be shuffled, you fucktrumpet'
    # gather all detections and annotations
    all_detections     = _get_detections(generator, model, distance=distance,score_threshold=score_threshold, max_detections=max_detections, save_path=save_path, render=render, workers=workers)
    all_annotations    = _get_annotations(generator)


    # all_detections = pickle.load(open('all_detections.pkl', 'rb'))
    # all_annotations = pickle.load(open('all_annotations.pkl', 'rb'))
    # pickle.dump(all_detections, open('all_detections.pkl', 'wb'))
    # pickle.dump(all_annotations, open('all_annotations.pkl', 'wb'))

    # process detections and annotations
    score_thresholds = np.arange(0.05,1.00,0.05)
    score_thresholds = np.append(score_thresholds, np.arange(0.05,0.1,0.01))
    score_thresholds = np.unique(np.sort(score_thresholds))

    best_map = 0.0
    for st in score_thresholds:
        loss_errors = np.zeros((0,))
        loss_errors_rel = np.zeros((0,))
        p_list = []
        r_list = []
        average_precisions = {}
        recall_dict            = {}
        precision_dict         = {}
        mean_dist_errors    = {}
        mean_dist_errors_rel    = {}
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            false_positives = np.zeros((0,))
            true_positives  = np.zeros((0,))
            scores          = np.zeros((0,))
            dist_errors     = np.zeros((0,))
            rel_dist_errors = np.zeros((0,))
            num_annotations = 0.0

            for i in range(generator.size()):
                detections           = all_detections[i][label]
                annotations          = all_annotations[i][label]
                num_annotations     += annotations.shape[0]
                detected_annotations = []

                for d in detections:
                    score = d[4]
                    if score < st:
                        continue
                    scores = np.append(scores,score)
                    if distance: dist = d[5]

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
                        if distance:
                            dist_err = np.abs(dist - annotations[assigned_annotation[0],4])
                            rel_dist_err = dist_err / annotations[assigned_annotation[0],4]
                            dist_errors = np.append(dist_errors, dist_err)
                            rel_dist_errors = np.append(rel_dist_errors, rel_dist_err)
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives  = np.append(true_positives, 0)


            # no annotations -> AP for this class is 0 (is this correct?)
            if num_annotations == 0:
                average_precisions[label] = 0, 0
                recall_dict[label] = 0
                precision_dict[label] = 0
                if distance: 
                    mean_dist_errors[label] = np.nan
                    mean_dist_errors_rel[label] = np.nan
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
            recall_dict[label] = np.mean(recall)
            precision_dict[label] = np.mean(precision)
            if distance: 
                mean_dist_errors[label] = np.mean(dist_errors)
                mean_dist_errors_rel[label] = np.mean(rel_dist_errors)


        # compute per class average precision
        total_instances = []
        precisions = []
        for label, (average_precision, num_annotations ) in average_precisions.items():
            total_instances.append(num_annotations)
            precisions.append(average_precision)
            if distance:
                loss_errors = np.append(loss_errors, mean_dist_errors[label])
                loss_errors_rel = np.append(loss_errors_rel, mean_dist_errors_rel[label])
            p_list.append(precision_dict[label])
            r_list.append(recall_dict[label])

        # ignore bg with [:-1]
        mean_precision = np.nanmean(p_list[:-1])
        mean_recall = np.nanmean(r_list[:-1])
        mean_ap = sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)
        if distance and np.count_nonzero(~np.isnan(loss_errors)):
            mean_loss_error = sum([a * b for a, b in zip(total_instances, loss_errors) if (b==b)]) / sum(~np.isnan(loss_errors)*total_instances)
            mean_loss_error_rel = sum([a * b for a, b in zip(total_instances, loss_errors_rel) if (b==b)]) / sum(~np.isnan(loss_errors_rel)*total_instances)
        else:
            mean_loss_error = np.nan
            mean_loss_error_rel = np.nan
        print('mAP @ scorethreshold {0:.2f}: {1:.4f} (precision: {2:.4f}, recall: {3:.4f}, mADE: {4:.4f}, mRDE: {5:.4f})'\
            .format(st, mean_ap, mean_precision, mean_recall, mean_loss_error, mean_loss_error_rel))
        


        if mean_ap >= best_map:
            best_map = mean_ap
            best_st = st
            best_aps = average_precisions
            best_mean_loss_errors = loss_errors
            best_mean_loss_errors_rel =loss_errors_rel
            best_precisions = p_list
            best_recalls = r_list



    print('='*60)
    
    if distance:
        return best_map, best_st, best_aps, best_precisions, best_recalls, best_mean_loss_errors, best_mean_loss_errors_rel
    else:
        return best_map, best_st, best_aps, best_precisions, best_recalls
