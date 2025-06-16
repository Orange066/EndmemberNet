# Preprocess Tutorial

This section provides the steps for preprocessing the dataset.  
Please first download the original data from the [Zenodo repository](https://doi.org/10.5281/zenodo.15471213) or [Hugging Face](https://huggingface.co/datasets/Orange066/Unmixing_RawData). Then, unzip the file and place the folder in the root directory of `EndmemberNet`.


```
detection/
    data/
        hyps/
        images/
        scripts/
        unmixing/
            AllMask/
            datasets-tif/
            fluorescence_time_data_tif/
            unmixing.yaml
```

## Data Augmentation using Weighted Synthesis
Run the following command:
```
cd preprocess
python weighted_synthesis.py
```
This will generate two folders in `detection/data/unmixing/`:  
- `AllData`: the augmented dataset.  
- `AllData_ori`: a visual representation of the original dataset.

## Generate Data for Detection Network Training
```
python create_detection_datasets.py
```
This will create four folders under `detection/data/unmixing/`:  
- `box_visualization`: visualizes the labels of each sample.  
- `images`, `labels`, and `txt`: used for training the detection network.

## Generate Data for Segmentation Network Training
```
python create_segmentation_datasets.py
```
This will produce the following folders under `detection/data/unmixing/`:  
- `AllData_ori_patch`  
- `AllData_ori_patch_resize`  
- `AllMask_patch`  
- `AllMask_patch_resize`  

`AllData_ori_patch_resize` and `AllMask_patch_resize` are the resized (128×128) versions of `AllData_ori_patch` and `AllMask_patch`, respectively. These are used for training the segmentation network.

## Generate Test Data
```
python fluorescence_time_data.py
python fluorescence_time_data_fast.py
```
This will generate two folders under `detection/data/unmixing/`:  
- `fluorescence_time_data`  
- `fluorescence_time_data_fast`  

These folders are used for testing.