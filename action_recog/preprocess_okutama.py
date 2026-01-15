import os
import torch
from pathlib import Path
from collections import defaultdict
from typing import List
import cv2
import numpy as np
from torchvision.models.video import (
    MViT_V1_B_Weights,
    MViT_V2_S_Weights,
    R3D_18_Weights,
    S3D_Weights,
    Swin3D_B_Weights,
    Swin3D_T_Weights,
    mvit_v1_b,
    mvit_v2_s,
    r3d_18,
    s3d,
    swin3d_b,
    swin3d_t,
)

model_name_to_model_and_weights = {
    "s3d": (s3d, S3D_Weights.DEFAULT),
    "r3d_18": (r3d_18, R3D_18_Weights.DEFAULT),
    "swin3d_t": (swin3d_t, Swin3D_T_Weights.DEFAULT),
    "swin3d_b": (swin3d_b, Swin3D_B_Weights.DEFAULT),
    "mvit_v1_b": (mvit_v1_b, MViT_V1_B_Weights.DEFAULT),
    "mvit_v2_s": (mvit_v2_s, MViT_V2_S_Weights.DEFAULT),
}



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



def read_labels(labels_folder, scene_idx, scale_x, scale_y):
    with open(os.path.join(labels_folder, f"{scene_idx}.txt"), "r") as f:
        lines = f.readlines()

    frame_dict = dict()
    for line in lines:
        # line : track id, xmin, ymin, xmax, ymax, frame, lost, occluded, generated, label ('Person'), actions
        s = line.split(" ")
        frame_idx = int(s[5])
        if frame_idx not in frame_dict:
            frame_dict[frame_idx] = dict()
            frame_dict[frame_idx]['track_ids'] = []
            frame_dict[frame_idx]['bboxs'] = []
            frame_dict[frame_idx]['actions'] = []
        frame_dict[frame_idx]['track_ids'].append(int(s[0]))
        frame_dict[frame_idx]['bboxs'].append([int(s[1])*scale_x,int(s[2])*scale_y,int(s[3])*scale_x,int(s[4])*scale_y])
        frame_dict[frame_idx]['actions'].append(s[10])
    return frame_dict

def create_label_folders(out_path, labels):
    print(f"Creating label folders in {out_path}")
    for label in labels:
        output_folder = out_path / label
        output_folder.mkdir(parents=True, exist_ok=True)
    return output_folder


def preprocess_crops_for_video_cls(crops: List[np.ndarray], 
                                    input_size: list = None, 
                                    weights: torch.Tensor = None) -> torch.Tensor:
    # Preprocess a list of crops for video classification.
    # Args:
    #     crops (List[np.ndarray]): List of crops to preprocess. Each crop should have dimensions (H, W, C)
    #     input_size (tuple, optional): The target input size for the model. Defaults to (224, 224).
    # Returns:
    #     torch.Tensor: Preprocessed crops as a tensor with dimensions (1, T, C, H, W).
    if input_size is None:
        input_size = [224, 224]
    from torchvision.transforms import v2

    transform = v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(input_size, antialias=True),
            v2.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
        ]
    )
    processed_crops = [transform(torch.from_numpy(crop).permute(2, 0, 1)) for crop in crops]
    return torch.stack(processed_crops).unsqueeze(0).permute(0, 2, 1, 3, 4)



train_drone1_morning_scenes = ['1.1.1', '1.1.2', '1.1.3', '1.1.4', '1.1.5', '1.1.6', '1.1.7', '1.1.10', '1.1.11'] 
train_drone1_noon_scenes = ['1.2.2', '1.2.4', '1.2.5', '1.2.6', '1.2.7', '1.2.8', '1.2.9', '1.2.11']
train_drone2_morning_scenes = ['2.1.1', '2.1.2', '2.1.3', '2.1.4', '2.1.5', '2.1.6', '2.1.7', '2.1.10']
train_drone2_noon_scenes = ['2.2.2', '2.2.4', '2.2.5', '2.2.6', '2.2.7', '2.2.8', '2.2.9', '2.2.11']
val_drone1_morning_scenes = ['1.1.8', '1.1.9']
val_drone1_noon_scenes = ['1.2.1', '1.2.3', '1.2.10']
val_drone2_morning_scenes = ['2.1.8', '2.1.9']
val_drone2_noon_scenes = ['2.2.1', '2.2.3', '2.2.10']   


start_end_scenes = {
    '1.1.1': (50, 2272),
    '1.1.3': (101, 1965),
    '1.1.5': (120, 1559),
    '1.1.6': (462, 2145),
    '1.1.10': (420, 1601),
    '2.1.1': (80, 1252),
    '2.1.2': (180, 1397),
    '2.1.3': (10, 2877),
    '2.1.4': (100, 2107),
    '2.1.4': (100, 2107),
    '1.2.11': (0, 1583),
    '2.2.2': (150, 1465), 
    '2.2.11': (0, 776)
}

## Frames includes highly accurate bounding boxes
clean_frame_idxs = {
    '1.1.1': [i for i in range(50, 531)] + [i for i in range(600, 891)],
}


subfolder = 'TestSetFrames'
# subfolder = 'TrainSetFrames'
root_path = f'/mnt/hdd/data/Okutama_Action/{subfolder}'
out_path = f'/mnt/hdd/data/Okutama_Action/ActionRecognition/{subfolder}'
labels_folder = f"/mnt/hdd/data/Okutama_Action/{subfolder}/Labels/MultiActionLabels/3840x2160"
# input_scenes = ['1.2.2', '2.2.2', '1.2.4', '2.2.4', '1.2.6', '1.2.8', '2.2.8', '1.2.9', '2.2.9', '1.2.11', '2.2.11']
# input_scenes = ['1.1.8', '1.1.9', '2.1.8', '2.1.9']
input_scenes = ['1.2.1', '1.2.3', '1.2.10', '2.2.1', '2.2.3', '2.2.10']
# input_scenes = ['1.2.3', '1.2.10', '2.2.1', '2.2.3', '2.2.10']
use_clean_frames = False # Remove the frames that have noisy bounding boxes
labels = ["standing","sitting","walking","running","lying"]

skip_frame = 4
WIDTH, HEIGHT = 3840, 2160
resizsed_width, resizsed_height = 1280, 720
scale_x, scale_y = resizsed_width / WIDTH, resizsed_height / HEIGHT

## Video classification parameters
num_video_sequence_samples = 16
crop_margin_percentage = 5
video_cls_overlap_ratio = 0.25 # "overlap ratio between video sequences"
video_classifier_model = "r3d_18"
_, weights = model_name_to_model_and_weights[video_classifier_model]

debug_vis = True
# Define root path
root_path = Path(root_path)
out_path = Path(out_path)

num_image = 0 
for scene_split in input_scenes:
    if scene_split in train_drone1_morning_scenes or scene_split in val_drone1_morning_scenes:
        drone, time = 'Drone1', 'Morning'
    elif scene_split in train_drone1_noon_scenes or scene_split in val_drone1_noon_scenes:
        drone, time = 'Drone1', 'Noon'
    elif scene_split in train_drone2_morning_scenes or scene_split in val_drone2_morning_scenes:
        drone, time = 'Drone2', 'Morning'
    elif scene_split in train_drone2_noon_scenes or scene_split in val_drone2_noon_scenes:  
        drone, time = 'Drone2', 'Noon'
    else:
        raise ValueError(f'Invalid scene split: {scene_split}')
    
    frame_dict = read_labels(labels_folder, scene_split, scale_x, scale_y)

    if scene_split in start_end_scenes.keys():
        start_idx, end_idx = start_end_scenes[scene_split]
        cut_start_end_frames = True
    else:
        cut_start_end_frames = False

    source_folder = root_path / drone / time / 'Extracted-Frames-1280x720' / scene_split
    output_folder = out_path / scene_split
    output_folder.mkdir(parents=True, exist_ok=True)
    debug_folder = output_folder / 'debug'
    debug_folder.mkdir(parents=True, exist_ok=True)

    create_label_folders(output_folder, labels)

    track_history = defaultdict(list)
    crops_to_infer = []

    # image_files = [file for file in source_folder.iterdir() if file.is_file() and (file.suffix.lower() in ['.jpg', '.png', '.jpeg'])]
    # sorted_image_files = sorted(image_files, key=lambda x: int(x.name[:-4]))
    frame_counter = 0
    tracklet_num = 0 
    for frame_idx in frame_dict:
        if cut_start_end_frames:
            if frame_idx < start_idx or frame_idx > end_idx:
                continue # skip the frames that are not in the range
        if use_clean_frames:
            if idx not in clean_frame_idxs[scene_split]:
                continue # skip the frames that have noisy bounding boxes

        frame_path = source_folder / f"{frame_idx:d}.jpg"
        frame = cv2.imread(frame_path)

        frame_counter += 1
        print(f"Processing frame {frame_counter}")

        boxes = frame_dict[frame_idx]['bboxs']
        track_ids = frame_dict[frame_idx]['track_ids']
        actions = frame_dict[frame_idx]['actions']


        if frame_counter % skip_frame == 0:
            crops_to_infer = []
            track_ids_to_infer = []
            actions_to_infer = []

        for box, track_id, action in zip(boxes, track_ids, actions):
            if frame_counter % skip_frame == 0:
                # frame.shape = (2160, 3840, 3) / box.shape = (4,)
                crop = crop_and_pad(frame, box, crop_margin_percentage)
                # crop.shape = (224, 224, 3)
                if debug_vis:
                    debug_path = debug_folder / f"{frame_idx:d}_{track_id:d}.jpg"
                    cv2.imwrite(debug_path, crop)

                track_history[track_id].append(crop)

            if len(track_history[track_id]) > num_video_sequence_samples:
                track_history[track_id].pop(0)

            if len(track_history[track_id]) == num_video_sequence_samples and frame_counter % skip_frame == 0:
                crops = preprocess_crops_for_video_cls(track_history[track_id], weights=weights)
                crops_to_infer.append(crops)
                track_ids_to_infer.append(track_id)
                actions_to_infer.append(action)

        if crops_to_infer and (
            frame_counter % int(num_video_sequence_samples * skip_frame * (1 - video_cls_overlap_ratio)) == 0
        ):
            # crops_to_infer[0].shape = 1,3,16,224,224
            crops_batch = torch.cat(crops_to_infer, dim=0) # crops_batch.shape = 2,3,16,224,224
            print(f"crops_batch shape: {crops_batch.shape}")    
            print(f"track_ids_to_infer: {track_ids_to_infer}")
            print(f"actions_to_infer: {actions_to_infer}")

            for idx in range(len(track_ids_to_infer)):
                tid = track_ids_to_infer[idx]
                gt = actions_to_infer[idx]
                crop = crops_batch[idx]
                _label_folder = gt.strip().lower()[1:-1]
                if _label_folder not in labels:
                    continue
                _label_path = output_folder / _label_folder
                _crop_path = _label_path / f"{tracklet_num:d}_{tid:d}.pt"
                torch.save(crop.clone(), _crop_path)
                tracklet_num += 1
