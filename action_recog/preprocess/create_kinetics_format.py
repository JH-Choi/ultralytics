'''
 created python file to the project by adding the line from .mydata import Mydata to the init.py file in the same folder.

The classids.json file contains a mapping of class names and ids. It will look like the following:

{"ClassA": 0, "ClassB": 1, "ClassC": 2}
The .csv files define which of the videos will be used for training, validation and inference testing, and which classes they reference.

They should look like the following:

/SlowFast/data/MyData/ClassA/ins.mp4 0
/SlowFast/data/MyData/ClassC/tak.mp4 2
'''
import os
import json
from glob import glob

### Parameters
label_dict = {"Carrying": 0, "Lying": 1, "Sitting": 2, "Standing": 3, "Walking": 4}
DATA_DICT = '/mnt/hdd/code/ARLproject/ultralytics/tmp'
SPLIT = 'test'
json_out = os.path.join(DATA_DICT, 'classids.json')

### Write classids.json
with open(json_out, 'w') as f:
    json.dump(label_dict, f)

writer = open(f'{DATA_DICT}/{SPLIT}.csv', 'w')

for label_key in label_dict.keys():
    lbl_folder = os.path.join(DATA_DICT, label_key)
    videos = glob(f'{lbl_folder}/*.mp4')

    for video in videos:
        # video_name = video.replace(DATA_DICT, '')[1:] # remove /   
        video_name = video
        writer.write(f'{video_name} {label_dict[label_key]}\n')


