from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

from keras_applications import get_submodules_from_kwargs
from keras_applications import imagenet_utils
from keras_applications.imagenet_utils import decode_predictions
from keras_applications.imagenet_utils import _obtain_input_shape
from keras_applications import vgg16
from keras.applications import keras_modules_injection
from keras.backend import concatenate, shape
from keras.layers import Lambda, Concatenate
import keras.backend as K
import keras

@keras_modules_injection
def custom(*args, **kwargs):
    return vggmax(*args, **kwargs)

preprocess_input = imagenet_utils.preprocess_input

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.1/'
                'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

def min_max_pool2d(x):
    max_x =  K.pool2d(x, pool_size=(2, 2), strides=(2, 2))
    min_x = min_pool2d(x)
    return K.concatenate([max_x, min_x], axis=3) # concatenate on channel

def min_max_pool2d_output_shape(input_shape):
    shape = list(input_shape)
    shape[1] = int(shape[1]/2)
    shape[2] = int(shape[2]/2)
    shape[3] *= 2
    return tuple(shape)

def min_pool2d(x, padding='valid'):
    max_val = K.max(x) + 1 # we gonna replace all zeros with that value
    if x._keras_shape[2] <= 20:
        padding = 'same'
    # replace all 0s with very high numbers
    is_zero = max_val * K.cast(K.equal(x,0), dtype=K.floatx())
    x = is_zero + x
    
    # execute pooling with 0s being replaced by a high number
    min_x = -keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding=padding)(-x)

    # depending on the value we either substract the zero replacement or not
    is_result_zero = max_val * K.cast(K.equal(min_x, max_val), dtype=K.floatx()) 
    min_x = min_x - is_result_zero

    return min_x # concatenate on channel

def min_pool2d_output_shape(input_shape):
    shape = list(input_shape)
    shape[1] = int(shape[1]/2)
    shape[2] = int(shape[2]/2)
    return tuple(shape)

def vggmax(include_top=True,
          weights='imagenet',
          input_tensor=None,
          input_shape=None,
          pooling=None,
          classes=1000,
          cfg=None,
          **kwargs):
    """Instantiates the VGG16 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)`
            (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 input channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        fusion_blocks: list of indexes giving the blocks where radar and image is 
            concatenated. Input Layer is targeted by 0.
    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        all_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            all_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            all_input = input_tensor
    
    ## Read config variables
    fusion_blocks = cfg.fusion_blocks


    ## Model
    
    # Seperate input
    if len(cfg.channels) > 3:
        image_input = Lambda(lambda x: x[:, :, :, :3], name='image_channels')(all_input)
        radar_input = Lambda(lambda x: x[:, :, :, 3:], name='radar_channels')(all_input)
    
    # Bock 0 Fusion
    if len(cfg.channels) > 3:
        if 0 in fusion_blocks:
            x = Concatenate(axis=3, name='concat_0')([image_input, radar_input])
        else:
            x = image_input
    else:
        x = all_input

    # Block 1 - Image
    x = layers.Conv2D(int(64 * cfg.network_width), (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(x)
    x = layers.Conv2D(int(64 * cfg.network_width), (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    if cfg.pooling == 'maxmin':
        x = Lambda(min_max_pool2d, name='block1_pool')(x)
    else:
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 1 - Radar
    if len(cfg.channels) > 3:
        if cfg.pooling == 'min':
            y = Lambda(min_pool2d,  name='rad_block1_pool')(radar_input)
        elif cfg.pooling == 'maxmin':
            y = Lambda(min_max_pool2d, name='rad_block1_pool')(radar_input)
        elif cfg.pooling == 'conv':
            y = layers.Conv2D(int(64 * cfg.network_width), (3, 3),
                            activation='relu',
                            padding='same',
                            strides=(2, 2),
                            name='rad_block1_pool')(radar_input)
        else:
            y = layers.MaxPooling2D((2, 2), strides=(2, 2), name='rad_block1_pool')(radar_input)
        
        ## Concatenate Block 1 Radar to image
        if 1 in fusion_blocks:
            x = Concatenate(axis=3, name='concat_1')([x, y])


    # Block 2
    x = layers.Conv2D(int(128*cfg.network_width), (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = layers.Conv2D(int(128*cfg.network_width), (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    if cfg.pooling == 'maxmin':
        x = Lambda(min_max_pool2d, name='block2_pool')(x)
    else:
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 2 - Radar
    if len(cfg.channels) > 3:
        if cfg.pooling == 'min':
            y = Lambda(min_pool2d,  name='rad_block2_pool')(y)
        elif cfg.pooling == 'maxmin':
            y = Lambda(min_max_pool2d, name='rad_block2_pool')(y)
        elif cfg.pooling == 'conv':
            y = layers.Conv2D(int(64 * cfg.network_width), (3, 3),
                            activation='relu',
                            padding='same',
                            strides=(2, 2),
                            name='rad_block2_pool')(y)
        else:
            y = layers.MaxPooling2D((2, 2), strides=(2, 2), name='rad_block2_pool')(y)
        
        ## Concatenate Block 2 Radar to image
        if 2 in fusion_blocks:
            x = Concatenate(axis=3, name='concat_2')([x, y])

    # Block 3
    x = layers.Conv2D(int(256*cfg.network_width), (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.Conv2D(int(256*cfg.network_width), (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = layers.Conv2D(int(256*cfg.network_width), (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    if cfg.pooling == 'maxmin':
        x = Lambda(min_max_pool2d, name='block3_pool')(x)
    else:
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 3 - Radar
    if len(cfg.channels) > 3:
        if cfg.pooling == 'min':
            y = Lambda(min_pool2d,  name='rad_block3_pool')(y)
        elif cfg.pooling == 'maxmin':
            y = Lambda(min_max_pool2d, name='rad_block3_pool')(y)
        elif cfg.pooling == 'conv':
            y = layers.Conv2D(int(64 * cfg.network_width), (3, 3),
                            activation='relu',
                            padding='same',
                            strides=(2, 2),
                            name='rad_block3_pool')(y)
        else:
            y = layers.MaxPooling2D((2, 2), strides=(2, 2), name='rad_block3_pool')(y)
        
        ## Concatenate Block 3 Radar to image
        if 3 in fusion_blocks:
            x = Concatenate(axis=3, name='concat_3')([x, y])


    # Block 4
    x = layers.Conv2D(int(512*cfg.network_width), (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = layers.Conv2D(int(512*cfg.network_width), (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = layers.Conv2D(int(512*cfg.network_width), (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
    if cfg.pooling == 'maxmin':
        x = Lambda(min_max_pool2d, name='block4_pool')(x)
    else:
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 4 - Radar
    if len(cfg.channels) > 3:
        if cfg.pooling == 'min':
            y = Lambda(min_pool2d,  name='rad_block4_pool')(y)
        elif cfg.pooling == 'maxmin':
            y = Lambda(min_max_pool2d, name='rad_block4_pool')(y)
        elif cfg.pooling == 'conv':
            y = layers.Conv2D(int(64 * cfg.network_width), (2, 2),
                            activation='relu',
                            padding='valid',
                            strides=(2, 2),
                            name='rad_block4_pool')(y)
        else:
            y = layers.MaxPooling2D((2, 2), strides=(2, 2), name='rad_block4_pool')(y)
        
        ## Concatenate Block 4 Radar to image
        if 4 in fusion_blocks:
            x = Concatenate(axis=3, name='concat_4')([x, y])

    # Block 5
    x = layers.Conv2D(int(512*cfg.network_width), (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = layers.Conv2D(int(512*cfg.network_width), (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = layers.Conv2D(int(512*cfg.network_width), (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Block 5 - Radar
    if len(cfg.channels) > 3:
        if cfg.pooling == 'min':
            y = Lambda(min_pool2d,  name='rad_block5_pool')(y)
        elif cfg.pooling == 'maxmin':
            y = Lambda(min_max_pool2d, name='rad_block5_pool')(y)
        elif cfg.pooling == 'conv':
            y = layers.Conv2D(int(64 * cfg.network_width), (2, 2),
                            activation='relu',
                            padding='valid',
                            strides=(2, 2),
                            name='rad_block5_pool')(y)
        else:
            y = layers.MaxPooling2D((2, 2), strides=(2, 2), name='rad_block5_pool')(y)
        
        ## Concatenate Block 5 Radar to image
        if 5 in fusion_blocks:
            x = Concatenate(axis=3, name='concat_5')([x, y])


    if include_top:
        # Classification block
        x = layers.Flatten(name='flatten')(x)
        x = layers.Dense(4096, activation='relu', name='fc1')(x)
        x = layers.Dense(4096, activation='relu', name='fc2')(x)
        x = layers.Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = all_input
    # Create model.
    model = models.Model(inputs, x, name='vgg16')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            weights_path = keras_utils.get_file(
                'vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                file_hash='64373286793e3c8b2b4e3219cbf3544b')
        else:
            weights_path = keras_utils.get_file(
                'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='6d6bbae143d832006294945121d1f1fc')
        model.load_weights(weights_path)
        if backend.backend() == 'theano':
            keras_utils.convert_all_kernels_in_model(model)
    elif weights is not None:
        model.load_weights(weights)

    return model