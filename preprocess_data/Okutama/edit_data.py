import os
from pathlib import Path
from tqdm import tqdm

def generate_path(data_path):
    image_path = data_path / 'images'
    mask_path = data_path / 'masks'
    label_path = data_path / 'labels'
    image_path.mkdir(parents=True, exist_ok=True)
    label_path.mkdir(parents=True, exist_ok=True)
    mask_path.mkdir(parents=True, exist_ok=True)
    return image_path, label_path, mask_path

input_path = '/mnt/hdd/data/Okutama_Action/yolov8/val_Drone1_Morning'
input_path = Path(input_path)
source_folder = input_path / 'images' 
source_label_folder = input_path / 'labels'
source_mask_folder = input_path / 'masks'

# Manually Set this indexes
# val Drone1 Morning
remove_idx = [x for x in range(428, 542)] + [x for x in range(981, 1082)] + [x for x in range(1327, 1442)] \
    + [x for x in range(2062, 2163)] + [x for x in range(2659, 2823)] + [x for x in range(3222, 3334)]
print(f'Number of images to remove: {len(remove_idx)}')


# os.system(f'mv {source_folder} {source_folder}_backup')
# os.system(f'mv {source_label_folder} {source_label_folder}_backup')
# os.system(f'mv {source_mask_folder} {source_mask_folder}_backup')

source_folder = input_path / 'images_backup'
source_label_folder = input_path / 'labels_backup'
source_mask_folder = input_path / 'masks_backup'

image_path, label_path, mask_path = generate_path(input_path)

image_files = [file for file in source_folder.iterdir() if file.is_file() and (file.suffix.lower() in ['.jpg', '.png', '.jpeg'])]
sorted_image_files = sorted(image_files, key=lambda x: int(x.name[:-4]))
num_images = len(sorted_image_files)
print(f'Number of images: {num_images}')


idx_name = 0 
for idx, img_fn in tqdm(enumerate(sorted_image_files)):
    img_idx = int(img_fn.name[:-4])
    if img_idx not in remove_idx:
        out_image_file = image_path / f'{idx_name:05d}.jpg'
        out_label_file = label_path / f'{idx_name:05d}.txt'
        out_mask_file = mask_path / f'{idx_name:05d}.jpg'
        os.system(f'cp {img_fn} {out_image_file}')
        os.system(f'cp {source_label_folder / img_fn.name[:-4]}.txt {out_label_file}')
        os.system(f'cp {source_mask_folder / img_fn.name} {out_mask_file}')
        idx_name += 1
