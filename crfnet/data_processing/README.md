## Data 
This folder contains subfolders for denoising, data fusion and data generation.

### Data Fusion
The data fusion of camera and radar data is applied by projecting the radar points as additional channels into the image plain.
In order to overcome the challenge of a missing dimension radar is projected as lines with predefined height (see config).
As a result, a new data type called Radar Augmented Image (RAI) is visualized below. Two different methods for projection are used for the TUM data set (fish-eye cameras) and the nuscenes data set (pinhole cameras).

![overall architecture](/images/RAI_TUM.png)

### Data Generation
Data generators are implemented for many data sets such as nuScenes, KITTI, COCO or VOC. The TUM data (camra) set uses a csv_generator. All data generators inherit from the same generator class.
The generator scripts for nuScenes and TUM data set (camra) can be executed for debugging.