import argparse
import time
from collections import defaultdict
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import os
import cv2
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt

from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device

"""Classifies videos using pretrained TorchVision models; see https://pytorch.org/vision/stable/."""

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

def print_log(log_txt, log_str):
    log_txt.write(log_str)
    print(log_str)

def crop_and_pad(frame, box, margin_percent):
    """Crop box with margin and take square crop from frame."""
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1

    # Add margin
    margin_x, margin_y = int(w * margin_percent / 100), int(h * margin_percent / 100)
    x1, y1 = max(0, x1 - margin_x), max(0, y1 - margin_y)
    x2, y2 = min(frame.shape[1], x2 + margin_x), min(frame.shape[0], y2 + margin_y)

    # # Take square crop from frame
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


def preprocess_crops_for_video_cls(crops: List[np.ndarray], 
                                    input_size: list = None, 
                                    device: str or torch.device = "", 
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
    return torch.stack(processed_crops).unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)


class TrackletDataset(Dataset):
    def __init__(self, root_dir, indexs, transform=None, labels=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        # self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(os.listdir(root_dir)))}
        self.class_to_idx = {}
        for idx, cls_name in enumerate(labels):
            self.class_to_idx[cls_name] = idx

        for cls in self.class_to_idx:
            for index in indexs:
                cls_path = os.path.join(root_dir, index, cls)
                for file in os.listdir(cls_path):
                    if file.endswith(".pt") or file.endswith(".npy"):
                        self.samples.append((os.path.join(cls_path, file), self.class_to_idx[cls]))
        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # Load pre-extracted tensor of shape (T, C, H, W)
        tracklet = torch.load(path) if path.endswith(".pt") else torch.from_numpy(np.load(path))

        if self.transform:
            # Apply transform to each frame (C, H, W) inside tracklet
            tracklet = torch.stack([self.transform(frame) for frame in tracklet])

        # Reorder to (C, T, H, W)
        tracklet = tracklet.permute(1, 0, 2, 3)
        return tracklet, label


# okutama actionlist
action_list = ['Calling', 'Carrying', 'Drinking', 'Hand Shaking',
                'Hugging', 'Lying', 'Pushing/Pulling', 'Reading',
                'Running', 'Sitting', 'Standing', 'Walking']

def main(opt):
    """Main function."""
    # run(**vars(opt))
    # weights (str): Path to the YOLO model weights. Defaults to "yolo11n.pt".
    # device (str): Device to run the model on. Use 'cuda' for NVIDIA GPU, 'mps' for Apple Silicon, or 'cpu'. Defaults to auto-detection.
    # source (str): Path to mp4 video file or YouTube URL. Defaults to a sample YouTube video.
    # output_path (Optional[str], optional): Path to save the output video. Defaults to None.
    # crop_margin_percentage (int, optional): Percentage of margin to add around detected objects. Defaults to 10.
    # num_video_sequence_samples (int, optional): Number of video frames to use for classification. Defaults to 8.
    # skip_frame (int, optional): Number of frames to skip between detections. Defaults to 4.
    # video_cls_overlap_ratio (float, optional): Overlap ratio between video sequences. Defaults to 0.25.
    # fp16 (bool, optional): Whether to use half-precision floating point. Defaults to False.
    # video_classifier_model (str, optional): Name or path of the video classifier model. Defaults to "microsoft/xclip-base-patch32".
    # labels (List[str], optional): List of labels for zero-shot classification. Defaults to predefined list.

    weights = opt.weights
    device = opt.device
    labels = opt.labels
    # opt.video_classifier_model = "r3d_18"
    opt.video_classifier_model = "swin3d_t"

    
    # Training configs
    epochs = 10
    batch_size = 8
    lr = 1e-4

    # Fix feature extractor
    # finetune only a few layers


    # fps 
    # frame_width, frame_height = 1920, 1080
    skip_frame = 1

    # num_video_sequence_samples = opt.num_video_sequence_samples
    crop_margin_percentage = 10
    num_video_sequence_samples = 16
    WIDTH, HEIGHT = 3840, 2160
    labels_folder = "/mnt/hdd/data/Okutama_Action/TestSetFrames/Labels/MultiActionLabels/3840x2160"
    root_folder = "/mnt/hdd/data/Okutama_Action/ActionRecognition/"
    # save_path = "./output/syn_noon_1_2_2"
    save_path = opt.output_path
    os.makedirs(save_path, exist_ok=True)
    log_txt = open(os.path.join(save_path, 'log.txt'), 'w')
    log_txt.write(f"video_classifier_model: {opt.video_classifier_model}\n")


    # test_morning_idxs = ['1.1.8', '1.1.9', '2.1.8', '2.1.9']
    # train_noon_idxs = ['1.2.2', '2.2.2']
    # train_noon_idxs = ['1.1.1']
    # test_morning_idxs = ['1.1.8']
    # test_noon_idxs = ['1.2.1', '1.2.3', '1.2.10', '2.2.1', '2.2.3', '2.2.10']
    # test_noon_idxs = ['1.2.2']
    # scene_idxs = train_noon_idxs
    train_noon_idxs = opt.train_idxs
    test_noon_idxs = opt.test_idxs
    train_noon_idxs = train_noon_idxs.split('/')
    test_noon_idxs = test_noon_idxs.split('/')

    if labels is None:
        labels = ["standing","sitting","walking","running","lying"]
    num_classes = len(labels)
    # Initialize models and device
    device = select_device(device)


    # Initialize
    model, weights = model_name_to_model_and_weights[opt.video_classifier_model]
    model = model(weights=weights)

    if opt.video_classifier_model == "r3d_18" or opt.video_classifier_model == "s3d":
        model.fc = nn.Linear(model.fc.in_features, num_classes)  # Replace classifier
        model = model.to(device)
        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze last layer (fully connected)
        for param in model.fc.parameters():
            param.requires_grad = True
    elif opt.video_classifier_model == "swin3d_t":
        model.head = nn.Linear(model.head.in_features, num_classes)  # Replace classifier
        model = model.to(device)
        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze last layer (fully connected)
        for param in model.head.parameters():
            param.requires_grad = True

        # # Freeze all layers
        # for param in model.parameters():
        #     param.requires_grad = False

        # # Unfreeze only the final classification layer
        # for param in model.blocks[-1].proj.parameters():
        #     param.requires_grad = True

    # fine-tune the last block as well
    # for name, param in model.named_parameters():
    #     if "layer4" in name or "fc" in name:
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False


    from torchvision.transforms import v2
    input_size = [224, 224]
    # Define Transforms
    # each crop have H,W,C
    transform = v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(input_size, antialias=True),
            v2.Normalize(mean=weights.DEFAULT.transforms().mean, 
                                std=weights.DEFAULT.transforms().std)
        ]
    )
    # processed_crops = [transform(torch.from_numpy(crop).permute(2, 0, 1)) for crop in crops]
    # return torch.stack(processed_crops).unsqueeze(0).permute(0, 2, 1, 3, 4).to(self.device)
    #     torch.Tensor: Preprocessed crops as a tensor with dimensions (1, T, C, H, W).

    print(f"Loading train set")
    train_root_folder = os.path.join(root_folder, 'TrainSetFrames')
    train_ds = TrackletDataset(train_root_folder, indexs=train_noon_idxs, transform=None, labels=labels)
    # syn_path = "/mnt/hdd/code/human_data_generation/xrfeitoria/output/S2_orig_Drone1_Noon_1_2_2/auto_Drone1_Noon_1_2_2_alti0/composite_w_shadow"
    # train_ds = TrackletDataset(syn_path, indexs=["actions"], transform=None, labels=labels)

    print(f"Loading val set")
    val_ds = TrackletDataset(os.path.join(root_folder, 'TestSetFrames'), indexs=test_noon_idxs, transform=None, labels=labels)

    log_txt.write(f"Loaded {len(train_ds)} samples\n")
    log_txt.write(f"Train set: {train_noon_idxs}\n")
    log_txt.write(f"train root folder: {train_root_folder}\n")
    log_txt.write(f"Loaded {len(val_ds)} samples\n")
    log_txt.write(f"Val set: {test_noon_idxs}\n")
    log_txt.write(f"val root folder: {os.path.join(root_folder, 'TestSetFrames')}\n")


    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)


    # Optimizer & Loss
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Draw loss graph
    loss_list = []
    acc_list = []

    # Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        step = 0
        for videos, labels in train_loader:
            videos = videos.to(device) # (B, T, C, H, W)
            labels = labels.to(device) # (B)
            videos = videos.permute(0, 2, 1, 3, 4) # (B, C, T, H, W)
            # print('videos.shape', videos.shape)
            # if step == 10:
            #     break

            outputs = model(videos)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            loss_list.append(running_loss / (step+1))
            acc_list.append(correct/total)
            print(f"Epoch {epoch+1}/{epochs} | Step {step+1}/{len(train_loader)} | Loss: {running_loss / (step+1):.4f} | Acc: {correct/total:.4f}")
            step += 1

        print_log(log_txt, f"Epoch {epoch+1}/{epochs} | Loss: {running_loss / (step+1):.4f} | Acc: {correct/total:.4f}\n")

    # Validation
    if val_loader:
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for videos, labels in tqdm(val_loader):
                videos, labels = videos.to(device), labels.to(device)
                videos = videos.permute(0, 2, 1, 3, 4) # (B, C, T, H, W)
                outputs = model(videos)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        print_log(log_txt, f"Val Accuracy: {100 * correct / total:.2f}%\n")

    torch.save(model.state_dict(), os.path.join(save_path, f'model_{epoch}.pt'))
    print_log(log_txt, f"Model saved to {save_path}\n")

    # load_weight = torch.load(os.path.join(save_path, f'model_{epoch}.pt'))
    # model.load_state_dict(load_weight)

    # Draw loss graph
    plt.plot(loss_list, label='Loss')
    plt.plot(acc_list, label='Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss_acc.png'))



def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolo11n.pt", help="ultralytics detector model path")
    parser.add_argument("--device", default="", help='cuda device, i.e. 0 or 0,1,2,3 or cpu/mps, "" for auto-detection')
    parser.add_argument(
        "--source",
        type=str,
        default="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        help="video file path or youtube URL",
    )
    parser.add_argument("--output-path", type=str, default="output_video.mp4", help="output video file path")
    parser.add_argument(
        "--crop-margin-percentage", type=int, default=10, help="percentage of margin to add around detected objects"
    )
    parser.add_argument(
        "--num-video-sequence-samples", type=int, default=8, help="number of video frames to use for classification"
    )
    parser.add_argument("--skip-frame", type=int, default=2, help="number of frames to skip between detections")
    parser.add_argument(
        "--video-cls-overlap-ratio", type=float, default=0.25, help="overlap ratio between video sequences"
    )
    parser.add_argument("--fp16", action="store_true", help="use FP16 for inference")
    parser.add_argument(
        "--video-classifier-model", type=str, default="microsoft/xclip-base-patch32", help="video classifier model name"
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        type=str,
        default=["standing","sitting","walking","running","lying"],
        help="labels for zero-shot video classification",
    )
    parser.add_argument(
        "--train-idxs",
        type=str,
        default=None,
        help="train idxs",
    )
    parser.add_argument(
        "--test-idxs",
        type=str,
        default=None,
        help="test idxs",
    )
    
    return parser.parse_args()

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)