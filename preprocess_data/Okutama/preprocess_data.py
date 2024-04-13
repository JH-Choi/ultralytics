import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from data_utils import read_raw_label, read_write_image
import pdb

input_path = '/mnt/hdd/data/Okutama_Action'
output_path = '/mnt/hdd/data/Okutama_Action/yolov8'
input_path = Path(input_path)
output_path = Path(output_path)

# Create output directory
output_path.mkdir(parents=True, exist_ok=True)

train_drone1_morning_scenes = ['1.1.1', '1.1.2', '1.1.3', '1.1.4', '1.1.5', '1.1.6', '1.1.7', '1.1.10', '1.1.11'] 
train_drone1_noon_scenes = ['1.2.2', '1.2.4', '1.2.5', '1.2.6', '1.2.7', '1.2.8', '1.2.9', '1.2.11']
train_drone2_morning_scenes = ['2.1.1', '2.1.2', '2.1.3', '2.1.4', '2.1.5', '2.1.6', '2.1.7', '2.1.10']
train_drone2_noon_scenes = ['2.2.2', '2.2.4', '2.2.5', '2.2.6', '2.2.7', '2.2.8', '2.2.9', '2.2.11']
val_drone1_morning_scenes = ['1.1.8', '1.1.9']
val_drone1_noon_scenes = ['1.2.1', '1.2.3', '1.2.10']
val_drone2_morning_scenes = ['2.1.8', '2.1.9']
val_drone2_noon_scenes = ['2.2.1', '2.2.3', '2.2.10']   
target_labels = {'Running': 0, 'Walking': 1, 'Lying': 2, 'Sitting': 3, "Standing": 4}

WIDTH, HEIGHT = 3840, 2160
resizsed_width, resizsed_height = 1280, 720
scale_x, scale_y = resizsed_width / WIDTH, resizsed_height / HEIGHT

# Create Training Data
training_data_path = output_path / 'train'
training_image_path = training_data_path / 'images'
training_mask_path = training_data_path / 'masks'
training_label_path = training_data_path / 'labels'
training_image_path.mkdir(parents=True, exist_ok=True)
training_label_path.mkdir(parents=True, exist_ok=True)
training_mask_path.mkdir(parents=True, exist_ok=True)

# training_image_files = []
num_images = 0 
traninig_label_files = []
for scene in train_drone1_morning_scenes:
    source_folder = input_path / 'TrainSetFrames/Drone1/Morning/Extracted-Frames-1280x720' / scene
    source_label_txt = input_path / 'TrainSetFrames/Labels/MultiActionLabels/3840x2160' / (scene+'.txt')
    image_files = [file for file in source_folder.iterdir() if file.is_file() and (file.suffix.lower() in ['.jpg', '.png', '.jpeg'])]
    sorted_image_files = sorted(image_files, key=lambda x: int(x.name[:-4]))

    # read raw label text file (e.g. 1.1.1.txt)
    bboxes = read_raw_label(source_label_txt, target_labels)

    # Remove label which does not have action label
    processed_image_files = []
    for img_fn in sorted_image_files:
        img_idx = int(img_fn.name[:-4])
        if img_idx in bboxes.keys():
            processed_image_files.append(img_fn)

    for idx, img_fn in tqdm(enumerate(processed_image_files), desc=f'TrainSetFrames/Drone1/Morning/{scene} Label=>{source_label_txt} # of images=>{len(processed_image_files)}'):
        img_idx = idx + num_images
        out_image_file = training_image_path / f'{img_idx:05d}.jpg'
        out_label_file = training_label_path / f'{img_idx:05d}.txt'
        out_mask_file = training_mask_path / f'{img_idx:05d}.jpg'
        f1 = open(out_label_file, "w")

        raw_img = cv2.imread(str(img_fn))
        raw_img = cv2.resize(raw_img, (resizsed_width, resizsed_height))
        cv2.imwrite(str(out_image_file), raw_img)

        label_ = bboxes[int(img_fn.name[:-4])]

        for (xmin, ymin, xmax, ymax, _, label) in label_:
            start_coord = (int(xmin*scale_x), int(ymin*scale_y)) # xmin, ymin
            end_coord = (int(xmax*scale_x), int(ymax*scale_y)) # xmax, ymax
            x_c = (xmin + ((xmax - xmin) / 2)) / WIDTH
            y_c = (ymin + ((ymax - ymin) / 2)) / HEIGHT
            w = (xmax - xmin) / WIDTH
            h = (ymax - ymin) / HEIGHT
            f1.write("%d %0.6f %0.6f %0.6f %0.6f\n" %(label,x_c,y_c,w,h))
            raw_img = cv2.rectangle(raw_img,start_coord,end_coord,(0,0,255),5)
        f1.close()
        # For debugging, save bbox screenshots
        cv2.imwrite(str(out_mask_file), raw_img)

    num_images += len(processed_image_files)