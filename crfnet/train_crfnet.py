#!/usr/bin/env python3

"""
Main script for CRF-Net Training with online evaluation outputs a model with weights and tensorboard logs.

Arguments:
    --config <PATH TO CONFIG FILE>

Built up on source:
https://github.com/fizyr/keras-retinanet
"""

### Imports ###
# Standard library imports
import argparse
import os
import sys
import copy
import traceback

# Third party imports
import keras
import keras.preprocessing.image
import tensorflow as tf

# Allow relative imports when being executed as script.
if __name__ == "__main__" and not __package__:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    import crfnet  # noqa: F401
    __package__ = "crfnet"

# Local application imports
from crfnet.model import losses
from crfnet.model import architectures
from crfnet.model.architectures.retinanet import retinanet_bbox
from crfnet.utils.callbacks import RedirectModel, Evaluate
from crfnet.utils.anchor import make_shapes_callback
from crfnet.utils.config import get_config
from crfnet.utils.keras_version import check_keras_version
from crfnet.utils.model import freeze as freeze_model
from crfnet.utils.helpers import makedirs, get_session
from crfnet.utils.anchor_parameters import AnchorParameters
from crfnet.data_processing.generator.crf_main_generator import create_generators


def model_with_weights(model, weights, skip_mismatch, config=None, num_classes=None):
    """ Load weights for model.

    :param model:           <keras.Model>       The model to load weights for
    :param weights:         <string>            Path to the weights file to load
    :param skip_mismatch:   <bool>              If True, skips layers whose shape of weights doesn't match with the model.

    :return model:          <keras.Model>       The model with loaded weights
    """

    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
        if len(config.channels) > 3:
            config.channels = [0,1,2]
            img_backbone = architectures.backbone('vgg16')
            ## get img weights
            # create img model
            img_model, _, _ = create_models(
                backbone_retinanet=img_backbone.retinanet,
                num_classes=num_classes,
                weights=weights,
                multi_gpu=0, 
                freeze_backbone=False,
                lr=config.learning_rate,
                inputs=(None,None,3),
                cfg=config,
                distance = config.distance_detection,
                distance_alpha = config.distance_alpha
            )

            img_model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
            # layers with mismatch
            if 'max' in config.network:
                layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
            else:
                layers = ['block1_conv1']
            for layer_name in layers:
                model_weights = model.get_layer(layer_name).get_weights()
                img_weights = img_model.get_layer(layer_name).get_weights()
                # [0] is weights
                model_weights[0][:,:,:img_weights[0].shape[2],:] = img_weights[0]
                # [1] is bias
                model_weights[1] = img_weights[1]
                model.get_layer(layer_name).set_weights(model_weights)
                print('Loaded available image weights for layer {}'.format(layer_name))
    return model


def create_models(backbone_retinanet, num_classes, weights, multi_gpu=0,
                  freeze_backbone=False, distance=False, distance_alpha=1.0, lr=1e-5, cfg=None, inputs=(None,None,3)):
    """ Creates three models (model, training_model, prediction_model).

    :param backbone_retinanet:      <func>              A function to call to create a retinanet model with a given backbone
    :param num_classes:             <int>               The number of classes to train
    :param weights:                 <keras.Weights>     The weights to load into the model
    :param multi_gpu:               <int>               The number of GPUs to use for training
    :param freeze_backbone:         <bool>              If True, disables learning for the backbone
    :param distance:                <bool>              If True, distance detection is enabled
    :param distance_alpha:          <float>             Weighted loss factor for distance loss
    :param lr:                      <float>             Learning rate for network training
    :param cfg:                     <Configuration>     Config class with config parameters
    :param inputs:                  <tuple>             Input shape for neural network

    :return model:                  <keras.Model>       The base model. This is also the model that is saved in snapshots.
    :return training_model:         <keras.Model>       The training model. If multi_gpu=0, this is identical to model.
    :return prediction_model:       <keras.Model>       The model wrapped with utility functions to perform object detection 
                                                        (applies regression values and performs NMS).
    """

    modifier = freeze_model if freeze_backbone else None

    # load anchor parameters, or pass None (so that defaults will be used)
    if 'small' in cfg.anchor_params:
        anchor_params = AnchorParameters.small
        num_anchors = AnchorParameters.small.num_anchors()
    else:
        anchor_params = None
        num_anchors   = None

    # Keras recommends initialising a multi-gpu model on the CPU to ease weight sharing, and to prevent OOM errors.
    # optionally wrap in a parallel model
    if multi_gpu > 1:
        from keras.utils import multi_gpu_model
        
        with tf.device('/cpu:0'):
            model = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier, inputs=inputs, distance=distance), weights=weights, skip_mismatch=True, config=copy.deepcopy(cfg), num_classes=num_classes)
        
        training_model = multi_gpu_model(model, gpus=multi_gpu)
    else:
        model          = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier, inputs=inputs, distance=distance, cfg=cfg), weights=weights, skip_mismatch=True, config=copy.deepcopy(cfg), num_classes=num_classes)
        training_model = model

    try:
        from keras.utils import plot_model
        # Write the keras model plot into a file
        plot_path = os.path.join(cfg.tb_logdir, cfg.model_name)
        makedirs(plot_path)
        plot_model(training_model, to_file=(os.path.join(plot_path, cfg.network) + '.png'), show_shapes=True)
    except Exception:
        # TODO: Catch the particular exceptions
        print(traceback.format_exc())
        print(sys.exc_info()[2])

    # make prediction model
    prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params, score_thresh_train=cfg.score_thresh_train, class_specific_filter=cfg.class_specific_nms)

    # compile model
    if distance:
        training_model.compile(
            loss={
                'regression'    : losses.smooth_l1(),
                'classification': losses.focal(),
                'distance'      : losses.smooth_l1(alpha=distance_alpha)
            },
            optimizer=keras.optimizers.adam(lr=lr, clipnorm=0.001)
        )
    else:
        training_model.compile(
            loss={
                'regression'    : losses.smooth_l1(),
                'classification': losses.focal(),
            },
            optimizer=keras.optimizers.adam(lr=lr, clipnorm=0.001)
        )

    return model, training_model, prediction_model


def create_callbacks(model, prediction_model, validation_generator, cfg):
    """ Creates the callbacks to use during training.

    :param model:                   <keras.Model>           The base model.
    :param prediction_model:        <keras.Model>           The model that should be used for validation.
    :param validation_generator:    <Generator>             The generator for creating validation data.
    :param cfg:                     <Configuration>         Config class with config parameters.

    :return callbacks:              <list>                  A list of callbacks used for training.
    """
    callbacks = []
    
    # Add progbar
    progbar_callback = keras.callbacks.ProgbarLogger(count_mode='steps', stateful_metrics=None)
    callbacks.append(progbar_callback)

    if cfg.tensorboard:
        tb_logdir = os.path.join(cfg.tb_logdir, cfg.model_name)
        makedirs(tb_logdir)
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir                = tb_logdir,
            histogram_freq         = 0,
            batch_size             = cfg.batchsize,
            write_graph            = True,
            write_grads            = False,
            write_images           = True,
            embeddings_freq        = 0,
            embeddings_layer_names = None,
            embeddings_metadata    = None
        )
        tensorboard_callback.set_model(model)

        callbacks.append(tensorboard_callback)
    else:
        tensorboard_callback = None


    if cfg.data_set == 'coco':
        from .utils.coco import CocoEval

        # use prediction model for evaluation
        evaluation = CocoEval(validation_generator, tensorboard=tensorboard_callback)
    else:
        save_path = None
        if cfg.save_val_img_path:
            save_path = cfg.save_val_img_path +cfg.model_name
            os.makedirs(save_path)
        evaluation = Evaluate(validation_generator, distance=cfg.distance_detection, tensorboard=tensorboard_callback,
                              weighted_average=cfg.weighted_map, render=False, save_path=save_path, workers=cfg.workers)
    evaluation = RedirectModel(evaluation, prediction_model)
    callbacks.append(evaluation)

    # save the model
    if cfg.save_model:
        # ensure directory created first; otherwise h5py will error after epoch.
        makedirs(cfg.save_model)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                cfg.save_model,
                '{model_name}.h5'.format(model_name=cfg.model_name)
            ),
            verbose=1,
            save_best_only=True,
            monitor="mAP",
            mode='max'
        )
        checkpoint = RedirectModel(checkpoint, model)
        callbacks.append(checkpoint)

    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor    = 'loss',
        factor     = 0.75,
        patience   = 2,
        verbose    = 1,
        mode       = 'auto',
        min_delta  = 0.0001,
        cooldown   = 0,
        min_lr     = 1e-6
    ))

    return callbacks



def main():

    FILE_DIRECTORY = os.path.dirname(os.path.abspath(__file__)) 

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=os.path.join(FILE_DIRECTORY,"configs/local.cfg"))
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError("ERROR: Config file \"%s\" not found"%(args.config))
    else:
        cfg = get_config(args.config)

    model_name = args.config.split('/')[-1] 
    model_name = model_name.split('.')[0]
    cfg.model_name = cfg.runtime + "_" + model_name

    assert cfg.inference is False, "You are running a training in inference mode. Please check your config!"

    # setting seed
    from .utils.helpers import initialize_seed
    # Set seed to compare trainings and exclude randomness
    initialize_seed(cfg.seed)

    # create object that stores backbone information
    backbone = architectures.backbone(cfg.network)

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu

    keras.backend.tensorflow_backend.set_session(get_session(cfg.gpu_mem_usage))

    # create the generators
    if 'nuscenes' in cfg.data_set:
        train_generator, validation_generator, test_generator, test_night_generator, test_rain_generator = create_generators(cfg, backbone)
    else:
        train_generator, validation_generator = create_generators(cfg, backbone)
    

    # create the model
    weights = None
    if cfg.load_model:
        print('Loading model, this may take a second...')
        model            = architectures.load_model(cfg.load_model, backbone_name=cfg.network)

        training_model = model
        prediction_model = retinanet_bbox(model=model, anchor_params=None, class_specific_filter=cfg.class_specific_nms)
    else:
        if cfg.pretrain_basenet:
            weights = backbone.download_imagenet()

        in_shape = (cfg.image_size[0], cfg.image_size[1], len(train_generator.channels))

        print('Creating model, this may take a second...')
        model, training_model, prediction_model = create_models(
            backbone_retinanet=backbone.retinanet,
            num_classes=train_generator.num_classes(),
            weights=weights,
            multi_gpu=0, 
            freeze_backbone=False,
            lr=cfg.learning_rate,
            inputs=in_shape,
            cfg=cfg,
            distance = cfg.distance_detection,
            distance_alpha = cfg.distance_alpha
        )

    # print model summary
    print(model.summary())
    print("Model Parameters: ", model.count_params())
    

    # this lets the generator compute backbone layer shapes using the actual backbone model
    if 'vgg' in cfg.network or 'densenet' in cfg.network:
        train_generator.compute_shapes = make_shapes_callback(model)
        if validation_generator:
            validation_generator.compute_shapes = train_generator.compute_shapes

    # create the callbacks
    callbacks = create_callbacks(
        model,
        prediction_model,
        validation_generator,
        cfg,
    )

    # Use multiprocessing if cpu_count > 0
    use_multiprocessing = cfg.workers > 0
    
    # class weights
    class_weights_labels={}
    if cfg.class_weights:
        class_weights_names = cfg.class_weights

        for key in class_weights_names.keys():
            class_weights_labels[train_generator.name_to_label(key)] = float(class_weights_names[key])
    
    # Print outputs
    print()
    print("="*60)
    print("\t\t##### Parameters #####")
    print("="*60)
    descr = cfg.get_description()
    descr = os.linesep.join([s for s in descr.splitlines() if s.strip()])
    print(descr)

    print()
    print("="*60)
    print("\t\t##### Start Training #####")
    print("="*60)


    ## Start training
    training_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=len(train_generator),
        epochs=cfg.epochs,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        verbose=1,
        callbacks=callbacks,
        workers=cfg.workers,
        use_multiprocessing=use_multiprocessing,
        class_weight=class_weights_labels
    )


    ## Evaluate on test data_set
    print("="*60)
    print("\t\t##### Evaluate Test Set #####")
    print("="*60)

    # Load best model
    best_model = keras.models.load_model(cfg.save_model + cfg.model_name + '.h5', custom_objects=backbone.custom_objects)
    # load anchor parameters, or pass None (so that defaults will be used)
    if 'small' in cfg.anchor_params:
        anchor_params = AnchorParameters.small
    else:
        anchor_params = None

    best_prediction_model = retinanet_bbox(model=best_model, anchor_params=anchor_params, class_specific_filter=False)

    # Evaluate
    from .utils.eval_test import evaluate_test_set
    evaluate_test_set(best_prediction_model, test_generator, cfg, mode='all', tensorboard=callbacks[1], verbose=1)
    print("="*60)
    print("\t##### Evaluate Test Set at Night #####")
    print("="*60)
    evaluate_test_set(best_prediction_model, test_night_generator, cfg, mode='night', tensorboard=callbacks[1], verbose=1)
    print("="*60)
    print("\t##### Evaluate Test Set at Rain #####")
    print("="*60)
    evaluate_test_set(best_prediction_model, test_rain_generator, cfg, mode='rain', tensorboard=callbacks[1], verbose=1)

    print("="*60)
    print("\t######## Finished successfully ########")
    print("="*60)

if __name__ == '__main__':
    main()
    
