'''
UCF-ARG Dataset: https://www.crcv.ucf.edu/data/UCF-ARG.php
'''
import os
import cv2
import xmltodict
import glob
from tqdm import tqdm   
from collections import OrderedDict
import pdb

# List of all activity names to:extract the correct XGTF `attributes`
CLASSES = ['boxing', 'carrying', 'clapping', 'digging', \
        'jogging', 'open-close trunk', 'running', 'throwing', \
        'walking', 'waving']

video_annots = {}
annot_folder = '/mnt/hdd/data/UCF-ARG/UCF-ARG_Evaluation'
search_pattern = os.path.join(annot_folder, '*.xgtf')

for annot_filepath_full in tqdm(sorted(glob.glob(search_pattern, recursive=True)),
                                desc="Parsing Annotations...", unit=" annots"):
    print("Annotation", annot_filepath_full)
    with open(annot_filepath_full, "r") as f:
        f_content = f.read()
        result = xmltodict.parse(f_content)
        config = result['viper']['config']['descriptor']
        vid_filename = os.path.basename(
            result['viper']['data']['sourcefile']['@filename'])
        print(vid_filename)
        objects_xml = result['viper']['data']['sourcefile']['object']
        #print("config", config)
        objects = []
        # print("objects", objects_xml)
        for ox in objects_xml:
            if ox['@name'] == "Person" or (ox['@name'] == "object"
                                           and ox['attribute'][0]['data:svalue']['@value']
                                           == "man"):
                bboxes = {}
                activities = {}
                for attribute in ox['attribute']:
                    print("Attribute", attribute['@name'])
                    # print("Attribute bbox", attribute)  
                    # print("Attribute bbox", attribute)['data:bbox']  
                    if attribute['@name'] in ("Location", "bounding_box"):
                        if 'data:bbox' in attribute.keys():
                            for bbox in attribute['data:bbox']:
                                if isinstance(bbox, dict):
                                    timespan = tuple(
                                        map(int, bbox['@framespan'].split(":")))
                                    for frame_nr in range(timespan[0], timespan[1]+1):
                                        bboxes[frame_nr] = {
                                            'x': int(bbox['@x']),
                                            'y': int(bbox['@y']),
                                            'w': int(bbox['@width']),
                                            'h': int(bbox['@height'])
                                        }
                    elif attribute['@name'] in ("Activity1"):
                        if 'data:lvalue' in attribute.keys():
                            if attribute['@name'] not in activities.keys():
                                activities[attribute['@name']] = {'framespan': [], 'value': []}

                            activity_instances = attribute['data:lvalue']
                            if not isinstance(activity_instances, list):
                                activity_instances = [activity_instances]
                            for ai in activity_instances:
                                timespan = tuple(
                                    map(int, ai['@framespan'].split(":")))
                                activities[attribute['@name']]['framespan'].append(timespan)
                                activities[attribute['@name']]['value'].append(ai['@value'])
                objects.append({
                    'id': ox['@id'],
                    # 'timespan': tuple(map(int, ox['@framespan'].split(":"))),
                    'bboxes': bboxes,
                    'activities': activities
                })

        if vid_filename not in video_annots.keys():
            video_annots[vid_filename] = []
        video_annots[vid_filename] = objects

        #print("data for ",vid_filename, len(objects), objects)

# video_dir_out = os.path.join(FOLDER_BASE, FOLDER_OUT)

for vid_name, vid_objs in tqdm(video_annots.items(), desc="Cutting Videos...", unit=" videos"):
    # print("Processing", vid_name)
    print(len(vid_objs))
    pdb.set_trace()