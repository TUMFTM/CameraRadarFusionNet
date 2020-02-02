## Configs
This config folder holds the configuration files that are needed for training, testing and evaluation.
A description of all configuration parameters if given below.

| Parameter | Description | Example / Range|
|---|---|---|
| **Path** |
|save_model | Path to save the model  | ./saved_models/ |
|load_model | Path to load a model to continue training on it| empty or ./saved_models/model1.h5 e.g. |
| **Data** |
| data_set | Whether nuscenes, camra or coco for the specific data sets | nuscenes |
| data_path | Path to the data set | ~/data/nuscenes |
| save_val_img_path | Path so save evaluated validation images after every epoch | True / False |
| n_sweeps | Number of radar time steps used | 1 - 26 |
| radar_projection_height | Height of projected radar lines in m | 0.01 -> points, 1000 -> "barcode", or meters in between |
| noise_filter_perfect | Perfect noise filter based on ground truth | True / False|
| radar_filter_dist | Filter radar data from a certain distance in m | 100 |
| scene_selection | Define validtion and test set| default or debug |
| **Tensorboard** |
| tensorboard | True if tensorboard logs should be saved | True/False |
| logdir | Path to save tensorboard log files | ./tb_logs/ |
| **Computing** |
| seed | Random seed to perform training | e.g. 0 |
| gpu | Integer that specifies the system's GPU to run the training on | e.g. 0 |
| gpu_mem_usage | Proportion (between 0 and 1)of GPU memory that should be used | e.g. 0.5 |
| workers | Number of threads for generating data during training and evaluation | e.g. 4 |
| **Preprocessing** |
| normalize_radar | True if radar data should be normalized | True/False |
| random_transform | True for extended data augmentation with rotation, shear etc. | True/False|
| sample_selection | True to exclude samples without any objects from training | True/False|
| only_radar_annotated | Only keep bounding boxes that have according radar points| 1 for nuScenes method, 2 for points_in_box method |
| noisy_image_method | Generate noisy image in data generator| poisson, gauss, s&p-perpixel, blurr|
| noise_factor | Degree of noise, highly depends on the method | e.g. 0.2 or 1e-4 |
| **Hyperparameters**|
| learning_rate | Learning rate for training | e.g. 1e-4 |
| batchsize | Batch size used for training | e.g. 1|
| epochs | Number of epochs for training | e.g. 50 |
| weighted_map | True if mAP should be calculated weighted | True/False |
| category_mapping | Categories (classes) to be used or merged | see default.cfg|
| class_weights | Class weights for imbalanced classes | see default.cfg |
| **CRF-Net** |
| channels | Input channels (RGB + Radar) according to nuscenes| e.g. 0,1,2,18 (encoding below)|
| image_height | Height of the input image in pixels | e.g. 360 |
| image_width | Width of the input image in pixels | e.g. 640 |
| dropout_radar | Chance that a sample has no radar data during training | 0 - 1 e.g. 0.2|
| dropout_image | Chance that a sample has no image data during training | 0 - 1 e.g. 0.2|
| network | Feature extractor network | e.g. vgg-max-fpn or resnet101 | 
| network width | Width factor of neural network to adapt number of kernels | e.g. 0.5 or 1.5 |
| pooling | Pooling in radar branch | max, min or conv |
| anchor params | default or small for different anchor sizes | default or small |
| pretrain_basenet | True if feature extractor should initialized with ImageNet weights | True/False |
| distance_detection | True if distances should be predicted (by an extra loss function) | True/False |
| distance_alpha | Weight factor for distance loss | e.g. 10 |
| class_specific_nms | True if NMS should be specific to classes | True/False |
| score_thresh_train | Score trehsold, from which detections count as positive | e.g. 0.05 |





### Radar Augmented Image Channels

| Channel ID | Description |
|---|---|
|0 | R-Channel (Image)|
|1 | G-Channel (Image)|
|2 | B-Channel (Image)|
|3 | dyn_prop|
|4 | id|
|5 | rcs|
|6 | vx|
|7 | vy|
|8 | vx_comp|
|9 | vy_comp |
|10 | is_quality_valid |
|11 | ambig_state |
|12 | x_rms |
|13 | y_rms |
|14 | invalid_state |
|15 | pdh0 |
|16 | vx_rms |
|17 | vy_rms |
|18 | distance |
