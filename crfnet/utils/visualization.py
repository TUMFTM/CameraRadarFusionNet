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

import cv2
import numpy as np

from .colors import label_color, tum_colors
import pprint




def visualize_predictions(predictions, image_data_vis, generator, dist=False, verbose=False, cfg=None):
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
        color = tum_colors[key] *255
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
    
    return image_data_vis

def draw_box(image, box, color, thickness=2):
    """ Draws a box on an image with a given color.

    # Arguments
        image     : The image to draw on.
        box       : A list of 4 elements (x1, y1, x2, y2).
        color     : The color of the box.
        thickness : The thickness of the lines to draw a box with.
    """
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)


def draw_caption(image, box, caption):
    """ Draws a caption above the box in an image.

    # Arguments
        image   : The image to draw on.
        box     : A list of 4 elements (x1, y1, x2, y2).
        caption : String containing the text to draw.
    """
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def draw_boxes(image, boxes, color, thickness=2):
    """ Draws boxes on an image with a given color.

    # Arguments
        image     : The image to draw on.
        boxes     : A [N, 4] matrix (x1, y1, x2, y2).
        color     : The color of the boxes.
        thickness : The thickness of the lines to draw boxes with.
    """
    for b in boxes:
        draw_box(image, b, color, thickness=thickness)


def draw_detections(image, boxes, scores, labels, dist=False, color=None, label_to_name=None, score_threshold=0.5):
    """ Draws detections in an image.

    # Arguments
        image           : The image to draw on.
        boxes           : A [N, 4] matrix (x1, y1, x2, y2).
        scores          : A list of N classification scores.
        labels          : A list of N labels.
        color           : The color of the boxes. By default the color from keras_retinanet.utils.colors.label_color will be used.
        label_to_name   : (optional) Functor for mapping a label to a name.
        score_threshold : Threshold used for determining what detections to draw.
    """
    selection = np.where(scores > score_threshold)[0]

    for i in selection:
        c = color if color is not None else label_color(labels[i])
        draw_box(image, boxes[i, :], color=c)

        # draw labels
        if dist is False:
            caption = (label_to_name(labels[i]) if label_to_name else labels[i]) + ': {0:.2f}'.format(scores[i])
        else:
            caption = (label_to_name(labels[i]) if label_to_name else labels[i]) + ': {0:.2f}   {1:.2f}m'.format(scores[i], dist[i])
        draw_caption(image, boxes[i, :], caption)


def draw_annotations(image, annotations, color=(0, 255, 0), label_to_name=None):
    """ Draws annotations in an image.

    # Arguments
        image         : The image to draw on.
        annotations   : A [N, 5] matrix (x1, y1, x2, y2, label) or dictionary containing bboxes (shaped [N, 4]) and labels (shaped [N]).
        color         : The color of the boxes. By default the color from keras_retinanet.utils.colors.label_color will be used.
        label_to_name : (optional) Functor for mapping a label to a name.
    """
    if isinstance(annotations, np.ndarray):
        annotations = {'bboxes': annotations[:, :4], 'labels': annotations[:, 4]}

    assert('bboxes' in annotations)
    assert('labels' in annotations)
    assert(annotations['bboxes'].shape[0] == annotations['labels'].shape[0])

    for i in range(annotations['bboxes'].shape[0]):
        label   = annotations['labels'][i]
        c       = color if color is not None else label_color(label)
        caption = '{}'.format(label_to_name(label) if label_to_name else label)
        draw_caption(image, annotations['bboxes'][i], caption)
        draw_box(image, annotations['bboxes'][i], color=c)
