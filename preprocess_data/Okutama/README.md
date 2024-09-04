### Data Structure 

```
Okutama_Action
├── TrainSetFrames
│   ├── Drone1
│   │   ├── Morning/Extracted-Frames-1280x720
|   │   │   ├── 1.1.1
|   │   │   ├── 1.1.2
|   │   │   ├── ....
│   │   ├── Noon/Extracted-Frames-1280x720
|   │   │   ├── 1.2.2
|   │   │   ├── 1.2.4
|   │   │   ├── ....
│   ├── Drone2
│   │   ├── Morning/Extracted-Frames-1280x720
|   │   │   ├── 2.1.1
|   │   │   ├── ....
│   │   ├── Noon/Extracted-Frames-1280x720
|   │   │   ├── 2.2.2
|   │   │   ├── ....
│   ├── Labels/MultiActionLabels/3840x2160
|   │   │   ├── 1.1.1.txt
|   │   │   ├── ....
|   │   │   ├── 2.1.1.txt
|   │   │   ├── ....
├── TestSetFrames
│   ├── Drone1
│   │   ├── Morning/Extracted-Frames-1280x720
|   │   │   ├── 1.1.8
|   │   │   ├── 1.1.9
│   │   ├── Noon/Extracted-Frames-1280x720
|   │   │   ├── 1.2.1
|   │   │   ├── 1.2.3
|   │   │   ├── 1.2.10
|   ├── Drone2
│   │   ├── Morning/Extracted-Frames-1280x720
|   │   │   ├── 2.1.8
|   │   │   ├── 2.1.9
│   │   ├── Noon/Extracted-Frames-1280x720
|   │   │   ├── 2.2.1
|   │   │   ├── 2.2.3
|   │   │   ├── 2.2.10
│   ├── Labels/MultiActionLabels/3840x2160
|   │   │   ├── 1.1.8.txt
|   │   │   ├── ....
|   │   │   ├── 2.1.8.txt
|   │   │   ├── ....
├
```

Label file format:
```
TrackID, xmin, ymin, xmax, ymax, frame, lost, occluded, generated, label, actions
```

### Official Paper
train-val set: 33 video sequences
test set: 10 video sequences

### Preprocess Detection
```
python preprocess_data_detection.py
```


### Preprocess Multi-Action Labels
```
python preprocess_data_multiaction.py
```


### TestSet  
* Drone1/Morning: 1.1.8, 1.1.9 => Drone is translating without rotation
* Drone1/Noon: 1.2.1, 1.2.3, 1.2.10 => Drone is orbiting
* Drone2/Morning: 2.1.8, 2.1.9 => Drone is stationary and rotating the camera
* Drone2/Noon : 2.2.10.mp4 => Drone is rising from the ground 