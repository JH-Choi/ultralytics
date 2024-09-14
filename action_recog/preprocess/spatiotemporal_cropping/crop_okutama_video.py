import cv2
from tqdm import tqdm
from pathlib import Path

from cutcrop_functions import print_if, cut_and_crop
from okutama_utils import read_raw_label
import pdb

input_path = '/mnt/hdd/data/Okutama_Action'
output_path = '/mnt/hdd/data/Okutama_Action/ActionRecognition'
ENABLE_CROP = True              # Set to True to enable Cropping to Bbox
VERBOSE_OUTPUT = False          # Set to True to enable DEBUG Messages
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

target_labels = {'Running': 0, 'Walking': 1, 'Lying': 2, 'Sitting': 3, "Standing": 4, 
                 'Calling': 5, 'Carrying': 6, 'Drinking': 7, 'Hand Shaking': 8,
                 'Hugging': 9, 'Pushing/Pulling': 10, 'Reading': 11}

WIDTH, HEIGHT = 3840, 2160
resizsed_width, resizsed_height = 1280, 720
crop_width, crop_height = 224, 224

def preprocess_data(num_images, scene_list, src_path, src_label_path):
    for scene in scene_list:
        video_filepath_in = str(input_path / src_path / (scene + '.mp4'))
        source_label_txt = input_path / src_label_path / (scene+'.txt')
        print('source_label_txt=>', source_label_txt)
        # image_files = [file for file in source_folder.iterdir() if file.is_file() and (file.suffix.lower() in ['.jpg', '.png', '.jpeg'])]
        # sorted_image_files = sorted(image_files, key=lambda x: int(x.name[:-4]))

        # read raw label text file (e.g. 1.1.1.txt)
        bboxes = read_raw_label(source_label_txt, target_labels)

        cap = cv2.VideoCapture(video_filepath_in)
        if not cap.isOpened():
            print("WARNING could not open video (skipping)", video_filepath_in)
            continue
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        input_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print("input_size=>", input_size)   

        for tracking_id in bboxes.keys():
            print("tracking_id=>", tracking_id)
            bbox_list = bboxes[tracking_id]
            activity_name = bbox_list[0][-1] 
            ts = bbox_list[0][4] # start frame
            te = bbox_list[-1][4] # end frame
            timespan = (ts, te)
            cut_and_crop(mode="ffmpeg", input_video_cap=cap, input_fps=fps, 
                        input_size=input_size, input_video_name=str(scene), 
                        dataset_name="okutama", activity_name=activity_name, 
                        activity_id=tracking_id, output_dir='./tmp', 
                        timespan=timespan, bbox=bbox_list, preview_video=False,
                        crop=ENABLE_CROP, verbose_output=VERBOSE_OUTPUT)
        


# Create train Data
# num_img = preprocess_data(0, train_drone1_morning_scenes, 
#                 src_path='TrainSetVideos/Drone1/Morning', 
#                 src_label_path='TrainSetVideos/Labels/MultiActionLabels/3840x2160')


num_img = preprocess_data(0, val_drone1_morning_scenes, 
                src_path='TestSetVideos/Drone1/Morning', 
                src_label_path='TestSetVideos/Labels/MultiActionLabels/3840x2160')

