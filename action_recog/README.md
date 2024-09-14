# Human Action Recognition
## Preprocess Data
### Okutama-Action 
crop_okutama_video.py : Ground Truth tracking ID is too noisy for cropping video


### UCF-Aerial 

## Training with SlowFast in pytorchvideo 


## Training with SlowFast
### Install SlowFast 
```bash
install slowfast.sh
```

### Preprocess UCF-ARG Data
- https://github.com/AlexanderMelde/SPHAR-Dataset/tree/master

### Preprocess Okutama-Action Data
- https://github.com/hitottiez/mht-paf
- https://github.com/DeepRobot2020/MobileDet/blob/master/datasets/okutama_to_hd5.py
- https://github.com/hitottiez/deepsort/tree/master?tab=readme-ov-file 


### Fine-tuning SlowFast in pytorchvideo
- https://github.com/facebookresearch/pytorchvideo/pull/41

### References
- https://gitlab.umiacs.umd.edu/dspcad/ptl-release/-/tree/main 
- https://github.com/Ricky-Xian/DARPA_TRAIN/blob/main/DARPA_TRAIN/src/train_video.py#L15
- https://github.com/ultralytics/ultralytics/issues/14461
- https://github.com/divyakraman/DIFFAR2022_DifferentiableFrequencyBasedDisentanglement/tree/main
- https://github.com/divyakraman/ECCV2022_FARFourierAerialVideoRecognition/blob/main/train1_uavhuman.py