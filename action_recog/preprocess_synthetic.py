import os
import json
import torch
from pathlib import Path
from collections import defaultdict
from typing import List
import cv2
import numpy as np

def crop_and_pad(frame, box, margin_percent):
    """
    Crop box with margin and take square crop from frame.
    """
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1

    # Add margin
    margin_x, margin_y = int(w * margin_percent / 100), int(h * margin_percent / 100)
    x1, y1 = max(0, x1 - margin_x), max(0, y1 - margin_y)
    x2, y2 = min(frame.shape[1], x2 + margin_x), min(frame.shape[0], y2 + margin_y)

    # Take square crop from frame
    # size = max(y2 - y1, x2 - x1)
    # center_y, center_x = (y1 + y2) // 2, (x1 + x2) // 2
    # half_size = size // 2
    # square_crop = frame[
    #     max(0, center_y - half_size) : min(frame.shape[0], center_y + half_size),
    #     max(0, center_x - half_size) : min(frame.shape[1], center_x + half_size),
    # ]
    # return cv2.resize(square_crop, (224, 224), interpolation=cv2.INTER_LINEAR)

    rectangle_crop = frame[y1:y2, x1:x2]
    return cv2.resize(rectangle_crop, (224, 224), interpolation=cv2.INTER_LINEAR)

root_path = '/mnt/hdd/code/human_data_generation/xrfeitoria/output/S2_Drone1_Noon_1_2_2/auto_Drone1_Noon_1_2_2_alti0'
root_path = Path(root_path)
image_folder = root_path / 'images'
label_folder = root_path / 'labels'
mask_folder = root_path / 'masks'

lbl_to_stencil_file = str(root_path / 'lbl_stencil.json')
with open(lbl_to_stencil_file, 'r') as json_file:
    label2stencil = json.load(json_file)




root_path