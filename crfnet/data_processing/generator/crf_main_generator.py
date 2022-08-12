from crfnet.utils.transform import random_transform_generator
from crfnet.utils.anchor_parameters import AnchorParameters
from crfnet.data_processing.generator.splits.nuscenes_splits import Scenes
from crfnet.utils.anchor_calc import anchor_targets_bbox
from crfnet.utils.anchor import guess_shapes

def create_generators(cfg, backbone):
  """ Create generators for training and validation and test data.

  :param cfg:                     <Configuration>         Config class with config parameters.
  :param backbone:                <Backbone>              Backbone class e.g. VGGBackbone

  :return train_generator:        <Generator>             The generator for creating training data.
  :return validation_generator:   <Generator>             The generator for creating validation data.

  TODO: @Max make the create generators consistently return train, val and test
  """
  if cfg.anchor_params:
    if 'small' in cfg.anchor_params:
      anchor_params = AnchorParameters.small
    else:
      anchor_params = None
  else:
    anchor_params = None

  common_args = {
    'batch_size': cfg.batchsize,
    'config': None,
    'image_min_side': cfg.image_size[0],
    'image_max_side': cfg.image_size[1],
    'filter_annotations_enabled': False,
    'preprocess_image': backbone.preprocess_image,
    'normalize_radar': cfg.normalize_radar,
    'camera_dropout': cfg.dropout_image,
    'radar_dropout': cfg.dropout_radar,
    'channels': cfg.channels,
    'distance': cfg.distance_detection,
    'sample_selection': cfg.sample_selection,
    'only_radar_annotated': cfg.only_radar_annotated,
    'n_sweeps': cfg.n_sweeps,
    'noise_filter': cfg.noise_filter_cfg,
    'noise_filter_threshold': cfg.noise_filter_threshold,
    'noisy_image_method': cfg.noisy_image_method,
    'noise_factor': cfg.noise_factor,
    'perfect_noise_filter': cfg.noise_filter_perfect,
    'radar_projection_height': cfg.radar_projection_height,
    'noise_category_selection': None if cfg.class_weights is None else cfg.class_weights.keys(),
    'inference': cfg.inference,
    'anchor_params': anchor_params,
  }

  # create random transform generator for augmenting training data
  if cfg.random_transform:
    transform_generator = random_transform_generator(
      min_rotation=-0.1,
      max_rotation=0.1,
      min_translation=(-0.1, -0.1),
      max_translation=(0.1, 0.1),
      min_shear=-0.1,
      max_shear=0.1,
      min_scaling=(0.9, 0.9),
      max_scaling=(1.1, 1.1),
      flip_x_chance=0.5,
      flip_y_chance=0.0,
    )
  else:
    transform_generator = random_transform_generator(flip_x_chance=0.5)

  category_mapping = cfg.category_mapping

  if 'nuscenes' in cfg.data_set:
    # import here to prevent unnecessary dependency on nuscenes
    from crfnet.data_processing.generator.nuscenes_generator import NuscenesGenerator
    from nuscenes.nuscenes import NuScenes

    if 'mini' in cfg.data_set:
      nusc = NuScenes(version='v1.0-mini', dataroot=cfg.data_path, verbose=True)
    else:
      try:
        nusc = NuScenes(version='v1.0-trainval', dataroot=cfg.data_path, verbose=True)
      except ValueError:
        nusc = NuScenes(version='v1.0-mini', dataroot=cfg.data_path, verbose=True)


    if 'debug' in cfg.scene_selection or 'mini' in cfg.data_set:
      scenes = Scenes.debug
    else:
      scenes = Scenes.default

    train_generator = NuscenesGenerator(
      nusc,
      scene_indices=scenes.train,
      transform_generator=transform_generator,
      category_mapping=category_mapping,
      compute_anchor_targets=anchor_targets_bbox,
      compute_shapes=guess_shapes,
      shuffle_groups=True,
      group_method='random',
      **common_args
    )

    # no dropouts in validation
    common_args['camera_dropout'] = 0
    common_args['radar_dropout'] = 0

    validation_generator = NuscenesGenerator(
      nusc,
      scene_indices=scenes.val,
      category_mapping=category_mapping,
      compute_anchor_targets=anchor_targets_bbox,
      compute_shapes=guess_shapes,
      **common_args
    )

    test_generator = NuscenesGenerator(
      nusc,
      scene_indices=scenes.test,
      category_mapping=category_mapping,
      compute_anchor_targets=anchor_targets_bbox,
      compute_shapes=guess_shapes,
      **common_args
    )

    test_night_generator = NuscenesGenerator(
      nusc,
      scene_indices=scenes.test_night,
      category_mapping=category_mapping,
      compute_anchor_targets=anchor_targets_bbox,
      compute_shapes=guess_shapes,
      **common_args
    )

    test_rain_generator = NuscenesGenerator(
      nusc,
      scene_indices=scenes.test_rain,
      category_mapping=category_mapping,
      compute_anchor_targets=anchor_targets_bbox,
      compute_shapes=guess_shapes,
      **common_args
    )
    return train_generator, validation_generator, test_generator, test_night_generator, test_rain_generator
  else:
    raise ValueError('Invalid data type received: {}'.format(cfg.data_set))