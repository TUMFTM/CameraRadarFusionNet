"""
Copyright 2017-2018 cgratie (https://github.com/cgratie/)

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

#todo: @Max purpose of vgg and vgg_max scripts? Unify scripts?
import keras
from keras.utils import get_file

from . import retinanet
from . import Backbone
from . import vggmax
from .vggmax import min_pool2d
from ...utils.image import preprocess_image

class VGGBackbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return vgg_retinanet(*args, backbone=self.backbone, **kwargs)

    def download_imagenet(self):
        """ Downloads ImageNet weights and returns path to weights file.
        Weights can be downloaded at https://github.com/fizyr/keras-models/releases .
        """
        if self.backbone == 'vgg16':
            resource = keras.applications.vgg16.vgg16.WEIGHTS_PATH_NO_TOP
            checksum = '6d6bbae143d832006294945121d1f1fc'
        elif 'vgg-max' in self.backbone:
            resource = keras.applications.vgg16.vgg16.WEIGHTS_PATH_NO_TOP
            checksum = '6d6bbae143d832006294945121d1f1fc'
        elif self.backbone == 'vgg19':
            resource = keras.applications.vgg19.vgg19.WEIGHTS_PATH_NO_TOP
            checksum = '253f8cb515780f3b799900260a226db6'
        else:
            raise ValueError("Backbone '{}' not recognized.".format(self.backbone))

        return get_file(
            '{}_weights_tf_dim_ordering_tf_kernels_notop.h5'.format(self.backbone),
            resource,
            cache_subdir='models',
            file_hash=checksum
        )

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['vgg16', 'vgg19', 'vgg-max', 'vgg-max-fpn']

        if self.backbone not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(self.backbone, allowed_backbones))

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return preprocess_image(inputs, mode='caffe')


def vgg_retinanet(num_classes, backbone='vgg16', inputs=None, modifier=None, distance=False, cfg=None, **kwargs):
    """ Constructs a retinanet model using a vgg backbone.

    Args
        num_classes: Number of classes to predict.
        backbone: Which backbone to use (one of ('vgg16', 'vgg19')).
        inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
        modifier: A function handler which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).

    Returns
        RetinaNet model with a VGG backbone.
    """
    # choose default input
    if inputs is None:
        inputs = keras.layers.Input(shape=(None, None, 3))
    elif isinstance(inputs, tuple):
        inputs = keras.layers.Input(inputs)
        
    # create the vgg backbone
    if backbone == 'vgg16':
        vgg = keras.applications.VGG16(input_tensor=inputs, include_top=False, weights=None)
    elif backbone == 'vgg19':
        vgg = keras.applications.VGG19(input_tensor=inputs, include_top=False, weights=None)
    elif 'vgg-max' in backbone:
        vgg = vggmax.custom(input_tensor=inputs, include_top=False, weights=None, cfg=cfg)
    else:
        raise ValueError("Backbone '{}' not recognized.".format(backbone))

    if modifier:
        vgg = modifier(vgg)

    # create the full model

    if 'max' in backbone and len(cfg.channels) >3:
        layer_names = []
        for i in range(3,6):
            if i in cfg.fusion_blocks:
                layer_names.append("concat_%i"%i)
            else:
                layer_names.append("block%i_pool"%i)
    else:
        layer_names = ["block3_pool", "block4_pool", "block5_pool"]

    layer_outputs = [vgg.get_layer(name).output for name in layer_names]

    radar_names = ["rad_block1_pool", "rad_block2_pool", "rad_block3_pool", "rad_block4_pool", "rad_block5_pool"]
    try:
        if 'fpn' in backbone:
            radar_outputs = [vgg.get_layer(name).output for name in radar_names]
            if cfg.pooling == 'min':
                radar_outputs.append(keras.layers.Lambda(min_pool2d, name='rad_block6_pool')(radar_outputs[-1]))
                radar_outputs.append(keras.layers.Lambda(min_pool2d, name='rad_block7_pool')(radar_outputs[-1]))
            elif cfg.pooling == 'conv':
                radar_outputs.append(keras.layers.Conv2D(int(64 * cfg.network_width), (3, 3),
                                activation='relu',
                                padding='same',
                                strides=(2, 2),
                                name='rad_block6_pool')(radar_outputs[-1]))
                radar_outputs.append(keras.layers.Conv2D(int(64 * cfg.network_width), (3, 3),
                                activation='relu',
                                padding='same',
                                strides=(2, 2),
                                name='rad_block7_pool')(radar_outputs[-1]))
            else:
                radar_outputs.append(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='rad_block6_pool',padding="same")(radar_outputs[-1]))
                radar_outputs.append(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='rad_block7_pool',padding="same")(radar_outputs[-1]))
        else:
            radar_outputs = None
    except Exception as e:
        radar_outputs = None
        # TODO: catch the specific exception. Exception is too broad.
        raise e

    return retinanet.retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=layer_outputs, radar_layers=radar_outputs, distance=distance, **kwargs)
