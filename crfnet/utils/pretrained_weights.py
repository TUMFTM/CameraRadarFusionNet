"""
This script provides the information of pretrained weights and downloads them.
They are safed into /.keras in the home folder and their path is passed as output.
As we load these models by name, we always include top, whatever this is.
"""

import keras.utils 

def get_weights_path(network):    
    
    if network == 'resnet':
        WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                        'releases/download/v0.2/'
                        'resnet50_weights_tf_dim_ordering_tf_kernels.h5')
        # WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
        #                     'releases/download/v0.2/'
        #                     'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')


        # Load weights.
        weights_path = keras.utils.get_file(
            'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
            WEIGHTS_PATH,
            cache_subdir='models',
            md5_hash='a7b3fe01876f51b976af0dea6bc144eb')

    elif 'vgg' in network:
        WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.1/'
                'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
        # WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
        #         'releases/download/v0.1/'
        #         'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

        weights_path = keras.utils.get_file(
                'vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                file_hash='64373286793e3c8b2b4e3219cbf3544b')

    elif network == 'inception_resnet':
        BASE_WEIGHT_URL = ('https://github.com/fchollet/deep-learning-models/'
                   'releases/download/v0.7/')
        
        fname = 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5'
        weights_path = keras.utils.get_file(
            fname,
            BASE_WEIGHT_URL + fname,
            cache_subdir='models',
            file_hash='e693bd0210a403b3192acc6073ad2e96')

    elif network == 'xception':
        TF_WEIGHTS_PATH = (
                'https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.4/'
                'xception_weights_tf_dim_ordering_tf_kernels.h5')
        weights_path = keras.utils.get_file(
                'xception_weights_tf_dim_ordering_tf_kernels.h5',
                TF_WEIGHTS_PATH,
                cache_subdir='models',
                file_hash='0a58e3b7378bc2990ea3b43d5981f1f6')

    else:
        # Raise Error
        print('No weights path found for this base net.')

    return weights_path