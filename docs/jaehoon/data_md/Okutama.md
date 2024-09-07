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

Original Labels: Each line contains 10+ columns, separated by spaces. The definition of these columns are:
- Track ID. All rows with the same ID belong to the same person for 180 frames. Then the person gets a new idea for the next 180 frames. We will soon release an update to make the IDs consistant.
- xmin. The top left x-coordinate of the bounding box.
- ymin. The top left y-coordinate of the bounding box.
- xmax. The bottom right x-coordinate of the bounding box.
- ymax. The bottom right y-coordinate of the bounding box.
- frame. The frame that this annotation represents.
- lost. If 1, the annotation is outside of the view screen.
- occluded. If 1, the annotation is occluded.
- generated. If 1, the annotation was automatically interpolated.
- label. The label for this annotation, enclosed in quotation marks.
- (+) actions. Each column after this is an action.

### Action List
```
action_list = ['Calling', 'Carrying', 'Drinking', 'Hand Shaking',
                'Hugging', 'Lying', 'Pushing/Pulling', 'Reading',
                'Running', 'Sitting', 'Standing', 'Walking']
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