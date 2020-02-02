# Standard Libraries
import numpy.random 
import tensorflow as tf
import random
import os

# 3rd Party Libraries
import cv2

def makedirs(path):
    """ Try to create the directory, pass if the directory exists already, fails otherwise.
    :param path:            <string>            directory path, that should be created

    """
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        if not os.path.isdir(path):
            raise

def layer_to_index(target_layer, model):
    """
    Turns the identifier of an layer into a 
    layer index, regardless if it is a str or already index

    :param target_layer: <str or int> identifier for the target layer. None will return None
    :param model: <keras.Model> The model containing the target layer

    :returns: <int> index of target layer
    """
    if target_layer is None:
        return None
    if isinstance(target_layer, str):
        for idx, layer in enumerate(model.layers):
            if layer.name == target_layer:
                return idx
    elif isinstance(target_layer, int):
        return target_layer
    else:
        raise TypeError("layer has to be int or str")

    raise Exception("Layer %s could not be found"%(str(target_layer)))

def tb_write_images(callback, names, imgs):
    """
    :param callback: <tensorflow.python.keras.callbacks.TensorBoard> Tensorboard callback
    :param names: <list of str> headings for the iamges
    :param imgs: <list of numpy.array> Images as Bitmaps
    """
    for name, img in zip(names, imgs):
        tf_img_enc = cv2.imencode('.jpg', img)[1].tostring()
        tf_img_tensor = tf.image.decode_jpeg(tf_img_enc, channels=3)
        tf_img_tensor = tf.expand_dims(tf_img_tensor, axis=0)

        imgsumarry=tf.summary.image(name, tf_img_tensor, max_outputs=3, collections=None)
        callback.writer.add_summary(imgsumarry.eval(session=tf.Session()))
        callback.writer.flush()

def tb_write_texts(callback, names, texts):
    """
    :param callback: <tensorflow.python.keras.callbacks.TensorBoard> Tensorboard callback
    :param names: <list of str> headings for the texts
    :param imgs: <list of str> Contents 
    """

    for name, text in zip(names, texts):
        txtsummary = tf.summary.text(name, tf.convert_to_tensor(text))
        callback.writer.add_summary(txtsummary.eval(session=tf.keras.backend.get_session()))
        callback.writer.flush()


def initialize_seed(seed=0):
    """
    This makes experiments more comparable by
    forcing the random number generator to produce
    the same numbers in each run
    """
    random.seed(a=seed)
    numpy.random.seed(seed)

    if hasattr(tf, 'set_random_seed'):
        tf.set_random_seed(seed)
    elif hasattr(tf.random, 'set_random_seed'):
        tf.random.set_random_seed(seed)
    elif hasattr(tf.random, 'set_seed'):
        tf.random.set_seed(seed)
    else: 
        raise AttributeError("Could not set seed for TensorFlow")


def get_session(gpu_usage=None):
    """ Construct a modified tf session.

    :param gpu_usage:       <float>             GPU memory usage from 0 to 1, None for dynamic growth

    :return tf.Session:     <tf.Session>        Tensorflow Session object
    """

    if gpu_usage:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_usage, \
        allow_growth=True)
    else:
        gpu_options = tf.GPUOptions(allow_growth=True)
        
    config = tf.ConfigProto(gpu_options=gpu_options)
    
    return tf.Session(config=config)


def output_index_by_name(model, output_name):
    """
    :param model: the keras model
    :param output_name: the string name for a specific output

    :returns: <int> specifying the index of the requested output
    """
    name_to_index = {name:i for i,name in enumerate(model.output_names)}
    return name_to_index[output_name]


def input_index_by_name(model, input_name):
    """
    :param model: the keras model
    :param output_name: the string name for a specific output

    :returns: <int> specifying the index of the requested output
    """
    name_to_index = {name:i for i,name in enumerate(model.input_names)}
    return name_to_index[input_name]