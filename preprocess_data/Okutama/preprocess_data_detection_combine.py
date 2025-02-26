import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from data_utils import read_raw_label, read_write_image
import pdb

input_path = '/mnt/hdd/data/Okutama_Action'
output_path = '/mnt/hdd/data/Okutama_Action/yolov8_Detection'
remove_noisy_label_for_test = True
# skip_interpolate_label = True
skip_interpolate_label = False

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

# Remove noisy labels for test set 
# Okutama-Action Data has many noisy lables => remove these noisy labels for test set
# 1.1.8: 301 ~ 376, 384 ~ 541, 873 ~ 902, 974~1083, 1184 ~1260, 1300~ 1441
# 1.1.9: 476 ~ 541, 606 ~ 720, 1051 ~ 1069, 1361~ 1440, 1452~ 1620, 1921~1980, 2040~2160, 2431~2488, 2624~2638
# 1.2.1: 461 ~ 509, 1081~1260, 
# 1.2.3: 0~9, 83~180, 224~239, 330~372, 442~540, 599~729, 801~829
# 1.2.10: X
# 2.1.8: 141 ~ 180, 230 ~ 360, 414 ~ 439, 471 ~ 720, 831~900, 1090 ~ 1260, 1411~1419, 1623~1800, 1961~1980, 
# 2.1.9: 201~219, 251~360, 451~540, 711~900, 921~1260, 1369~1620, 1691~1813
# 2.2.1: 277~360, 460~540, 1017~1260
# 2.2.3: 444~540, 560~721, 739~900, 1842~1921
# 2.2.10: 1059~1080, 1118~1129, 1281~1312, 1501~1589, 1671~1809, 

val_noisy_labels = {
    '1.1.1': [],
    '1.2.2': [],
    '1.1.8': [i for i in range(301, 377)] + [i for i in range(384, 542)] + [i for i in range(873, 903)] + [i for i in range(974, 1084)] + [i for i in range(1184, 1261)] + [i for i in range(1300, 1442)],
    '1.1.9': [i for i in range(476, 542)] + [i for i in range(606, 721)] + [i for i in range(1051, 1070)] + [i for i in range(1361, 1441)] + [i for i in range(1452, 1621)] + [i for i in range(1921, 1981)] + [i for i in range(2040, 2161)] + [i for i in range(2431, 2489)] + [i for i in range(2624, 2639)],
    '1.2.1': [i for i in range(461, 510)] + [i for i in range(1081, 1261)],
    '1.2.3': [i for i in range(0, 10)] + [i for i in range(83, 181)] + [i for i in range(224, 240)] + [i for i in range(330, 373)] + [i for i in range(442, 541)] + [i for i in range(599, 730)] + [i for i in range(801, 830)],
    '1.2.10': [],
    '2.1.8': [i for i in range(141, 181)] + [i for i in range(230, 361)] + [i for i in range(414, 440)] + [i for i in range(471, 721)] + [i for i in range(831, 901)] + [i for i in range(1090, 1261)] + [i for i in range(1411, 1420)] + [i for i in range(1623, 1801)] + [i for i in range(1961, 1981)],
    '2.1.9': [i for i in range(201, 220)] + [i for i in range(251, 361)] + [i for i in range(451, 541)] + [i for i in range(711, 901)] + [i for i in range(921, 1261)] + [i for i in range(1369, 1621)] + [i for i in range(1691, 1814)],
    '2.2.1': [i for i in range(277, 361)] + [i for i in range(460, 541)] + [i for i in range(1017, 1261)],
    '2.2.2': [],
    '2.2.3': [i for i in range(444, 541)] + [i for i in range(560, 722)] + [i for i in range(739, 901)] + [i for i in range(1842, 1922)],
    '2.2.10': [i for i in range(1059, 1081)] + [i for i in range(1118, 1130)] + [i for i in range(1281, 1313)] + [i for i in range(1501, 1590)] + [i for i in range(1671, 1810)]
}



target_labels = {'Running': 0, 'Walking': 1, 'Lying': 2, 'Sitting': 3, "Standing": 4}

WIDTH, HEIGHT = 3840, 2160
resizsed_width, resizsed_height = 1280, 720
scale_x, scale_y = resizsed_width / WIDTH, resizsed_height / HEIGHT

def generate_path(output_path, split):
    data_path = output_path / split
    image_path = data_path / 'images'
    mask_path = data_path / 'masks'
    label_path = data_path / 'labels'
    image_path.mkdir(parents=True, exist_ok=True)
    label_path.mkdir(parents=True, exist_ok=True)
    mask_path.mkdir(parents=True, exist_ok=True)
    return image_path, label_path, mask_path

def preprocess_data(num_images, scene_list, 
                    image_path, label_path, mask_path):
    '''
    Args:
        num_images: int, starting index for image numbering
        scene_list: list, list of scene names
        src_label_path: str, source label path
        image_path: str, output image path
        label_path: str, output label path
        mask_path: str, output mask path
    '''

    for scene in scene_list:
        if scene in train_drone1_morning_scenes:
            src_path = 'TrainSetFrames/Drone1/Morning/Extracted-Frames-1280x720'
            src_label_path='TrainSetFrames/Labels/MultiActionLabels/3840x2160'
        elif scene in train_drone1_noon_scenes:
            src_path = 'TrainSetFrames/Drone1/Noon/Extracted-Frames-1280x720'
            src_label_path='TrainSetFrames/Labels/MultiActionLabels/3840x2160'
        elif scene in train_drone2_morning_scenes:
            src_path = 'TrainSetFrames/Drone2/Morning/Extracted-Frames-1280x720'
            src_label_path='TrainSetFrames/Labels/MultiActionLabels/3840x2160'
        elif scene in train_drone2_noon_scenes:
            src_path = 'TrainSetFrames/Drone2/Noon/Extracted-Frames-1280x720'
            src_label_path='TrainSetFrames/Labels/MultiActionLabels/3840x2160' 
        elif scene in val_drone1_morning_scenes:
            src_path = 'TestSetFrames/Drone1/Morning/Extracted-Frames-1280x720'
            src_label_path='TestSetFrames/Labels/MultiActionLabels/3840x2160' 
        elif scene in val_drone1_noon_scenes:
            src_path = 'TestSetFrames/Drone1/Noon/Extracted-Frames-1280x720'
            src_label_path='TestSetFrames/Labels/MultiActionLabels/3840x2160' 
        elif scene in val_drone2_morning_scenes:
            src_path = 'TestSetFrames/Drone2/Morning/Extracted-Frames-1280x720'
            src_label_path='TestSetFrames/Labels/MultiActionLabels/3840x2160' 
        elif scene in val_drone2_noon_scenes:
            src_path = 'TestSetFrames/Drone2/Noon/Extracted-Frames-1280x720'
            src_label_path='TestSetFrames/Labels/MultiActionLabels/3840x2160' 

        source_folder = input_path / src_path / scene
        source_label_txt = input_path / src_label_path / (scene+'.txt')
        image_files = [file for file in source_folder.iterdir() if file.is_file() and (file.suffix.lower() in ['.jpg', '.png', '.jpeg'])]
        sorted_image_files = sorted(image_files, key=lambda x: int(x.name[:-4]))

        # read raw label text file (e.g. 1.1.1.txt)
        bboxes = read_raw_label(source_label_txt, target_labels, skip_interpolate=skip_interpolate_label)

        # Remove label which does not have action label
        processed_image_files = []
        for img_fn in sorted_image_files:
            img_idx = int(img_fn.name[:-4])

            # Skip noisy labels
            if img_idx in val_noisy_labels[scene] and remove_noisy_label_for_test:
                continue

            if img_idx in bboxes.keys():
                processed_image_files.append(img_fn)

        # for idx, img_fn in tqdm(enumerate(processed_image_files), desc=f'{src_path}/{scene} Label=>{source_label_txt} # of images=>{len(processed_image_files)}'):
        for idx, img_fn in tqdm(enumerate(processed_image_files)):
            img_idx = idx + num_images
            out_image_file = image_path / f'{img_idx:05d}.jpg'
            out_label_file = label_path / f'{img_idx:05d}.txt'
            out_mask_file = mask_path / f'{img_idx:05d}.jpg'
            # out_mask_file = mask_path / f'{img_fn.name}' # For Debugging

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

                label = 0  # Convert Every Action Labels to Person 

                f1.write("%d %0.6f %0.6f %0.6f %0.6f\n" %(label,x_c,y_c,w,h))
                raw_img = cv2.rectangle(raw_img,start_coord,end_coord,(0,0,255),5)

            f1.close()
            # For debugging, save bbox screenshots
            cv2.imwrite(str(out_mask_file), raw_img)

        num_images += len(processed_image_files)
    return num_images

#######################################
### Create train Data for each 
#######################################
# image_path, label_path, mask_path = generate_path(output_path, '1.2.2')
# num_img = preprocess_data(0, ['1.2.2'], 
#                 image_path=image_path, label_path=label_path, mask_path=mask_path)
# print('------------------------------------------')
# print('Train 1.2.2 Noon=>', num_img)
# print('------------------------------------------')

# image_path, label_path, mask_path = generate_path(output_path, '2.2.2')
# num_img = preprocess_data(0, ['2.2.2'], 
#                 image_path=image_path, label_path=label_path, mask_path=mask_path)
# print('------------------------------------------')
# print('Train 2.2.2 Noon=>', num_img)
# print('------------------------------------------')


image_path, label_path, mask_path = generate_path(output_path, '1.1.1')
num_img = preprocess_data(0, ['1.1.1'], 
                image_path=image_path, label_path=label_path, mask_path=mask_path)
print('------------------------------------------')
print('Train 1.1.1 Morning=>', num_img)
print('------------------------------------------')



#######################################
### Create train Data
#######################################
# image_path, label_path, mask_path = generate_path(output_path, 'train_Drone1_Morning')
# num_img = preprocess_data(0, train_drone1_morning_scenes, 
#                 image_path=image_path, label_path=label_path, mask_path=mask_path)
# print('------------------------------------------')
# print('Train Drone1 Morning=>', num_img)
# print('------------------------------------------')

# image_path, label_path, mask_path = generate_path(output_path, 'train_Drone1_Noon')
# num_img = preprocess_data(0, train_drone1_noon_scenes, 
#                 image_path=image_path, label_path=label_path, mask_path=mask_path)
# print('------------------------------------------')
# print('Train Drone1 Noon=>', num_img)
# print('------------------------------------------')

# image_path, label_path, mask_path = generate_path(output_path, 'train_Drone2_Morning')
# num_img = preprocess_data(0, train_drone2_morning_scenes, 
#                 image_path=image_path, label_path=label_path, mask_path=mask_path)
# print('------------------------------------------')
# print('Train Drone2 Morning=>', num_img)
# print('------------------------------------------')

# image_path, label_path, mask_path = generate_path(output_path, 'train_Drone2_Noon')
# num_img = preprocess_data(0, train_drone2_noon_scenes, 
#                 image_path=image_path, label_path=label_path, mask_path=mask_path)
# print('------------------------------------------')
# print('Train Drone2 Noon=>', num_img)
# print('------------------------------------------')

# image_path, label_path, mask_path = generate_path(output_path, 'train_D1D2_Morning')
# num_img = preprocess_data(0, train_drone1_morning_scenes + train_drone2_morning_scenes, 
#                 image_path=image_path, label_path=label_path, mask_path=mask_path)
# print('------------------------------------------')
# print('Train Drone1 + Drone2 Morning =>', num_img)
# print('------------------------------------------')


# image_path, label_path, mask_path = generate_path(output_path, 'train_D1D2_Noon')
# num_img = preprocess_data(0, train_drone1_noon_scenes + train_drone2_noon_scenes, 
#                 image_path=image_path, label_path=label_path, mask_path=mask_path)
# print('------------------------------------------')
# print('Train Drone1 + Drone2 Noon =>', num_img)
# print('------------------------------------------')

# image_path, label_path, mask_path = generate_path(output_path, 'train_all')
# num_img = preprocess_data(0, train_drone1_morning_scenes + train_drone2_morning_scenes + train_drone1_noon_scenes + train_drone2_noon_scenes, 
#                 image_path=image_path, label_path=label_path, mask_path=mask_path)
# print('------------------------------------------')
# print('Train All =>', num_img)
#

#######################################
### Create val Data
#######################################
# image_path, label_path, mask_path = generate_path(output_path, 'val_Drone1_Morning')

# num_img = preprocess_data(0, val_drone1_morning_scenes, 
#                 image_path=image_path, label_path=label_path, mask_path=mask_path)
# print('------------------------------------------')
# print('Test Drone1 Morning=>', num_img)
# print('------------------------------------------')


# image_path, label_path, mask_path = generate_path(output_path, 'val_Drone1_Noon')
# num_img = preprocess_data(0, val_drone1_noon_scenes, 
#                 image_path=image_path, label_path=label_path, mask_path=mask_path)
# print('------------------------------------------')
# print('Test Drone1 Noon=>', num_img)
# print('------------------------------------------')

# image_path, label_path, mask_path = generate_path(output_path, 'val_Drone2_Morning')
# num_img = preprocess_data(0, val_drone2_morning_scenes, 
#                 image_path=image_path, label_path=label_path, mask_path=mask_path)
# print('------------------------------------------')
# print('Test Drone2 Morning=>', num_img)
# print('------------------------------------------')

# image_path, label_path, mask_path = generate_path(output_path, 'val_Drone2_Noon')
# num_img = preprocess_data(0, val_drone2_noon_scenes, 
#                 image_path=image_path, label_path=label_path, mask_path=mask_path)
# print('------------------------------------------')
# print('Test Drone2 Noon=>', num_img)
# print('------------------------------------------')


# image_path, label_path, mask_path = generate_path(output_path, 'val_D1D2_Morning')
# num_img = preprocess_data(0, val_drone1_morning_scenes + val_drone2_morning_scenes, 
#                 image_path=image_path, label_path=label_path, mask_path=mask_path)
# print('------------------------------------------')
# print('Val Drone1 + Drone2 Morning =>', num_img)
# print('------------------------------------------')


# image_path, label_path, mask_path = generate_path(output_path, 'val_D1D2_Noon')
# num_img = preprocess_data(0, val_drone1_noon_scenes + val_drone2_noon_scenes, 
#                 image_path=image_path, label_path=label_path, mask_path=mask_path)
# print('------------------------------------------')
# print('Val Drone1 + Drone2 Noon =>', num_img)
# print('------------------------------------------')


# image_path, label_path, mask_path = generate_path(output_path, 'val_all')
# num_img = preprocess_data(0, val_drone1_morning_scenes + val_drone2_morning_scenes + val_drone1_noon_scenes + val_drone2_noon_scenes, 
#                 image_path=image_path, label_path=label_path, mask_path=mask_path)
# print('------------------------------------------')
# print('Val All =>', num_img)
# print('------------------------------------------')

