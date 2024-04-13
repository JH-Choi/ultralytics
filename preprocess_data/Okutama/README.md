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