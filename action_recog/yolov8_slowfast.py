'''
Written by Jaehoon
Code borrowed from examples/YOLOv8-Action-Recognition/action_recognition.py
Combine YOLOv8 and pyslowfast
'''

import argparse
import time
from collections import defaultdict
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import cv2
import json
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from ultralytics import YOLO
from ultralytics.data.loaders import get_best_youtube_url
from ultralytics.utils.plotting import Annotator
from ultralytics.utils.torch_utils import select_device

### Load SlowFast
# import os, sys 
# sys.path.append(os.path.join(os.getcwd(), "SlowFast"))

from slowfast.models import build_model
from slowfast.config.defaults import get_cfg
import slowfast.utils.checkpoint as cu

import pdb

class SlowfastVideoClassifier:

    def __init__(self, model_name: str, device: str or torch.device = ""):
        """
        Initialize the VideoClassifier with the specified model name and device.

        Args:
            model_name (str): The name of the model to use.
            device (str or torch.device, optional): The device to run the model on. Defaults to "".

        Raises:
            ValueError: If an invalid model name is provided.
        """
        # if model_name not in self.model_name_to_model_and_weights:
        #     raise ValueError(f"Invalid model name '{model_name}'. Available models: {self.available_model_names()}")
        # model, self.weights = self.model_name_to_model_and_weights[model_name]
        # self.device = select_device(device)
        # # self.model = build_model(cfg, )
        # # self.model = model(weights=self.weights).to(self.device).eval()


        # json_filename = "kinetics_classnames.json"
        # with open(json_filename, "r") as f:
        #     kinetics_classnames = json.load(f)
        # # Create an id to label name mapping
        # self.kinetics_id_to_classname = {}
        # for k, v in kinetics_classnames.items():
        #     self.kinetics_id_to_classname[v] = str(k).replace('"', "")

        self.device = select_device(device)

        # cfg = load_config()
        cfg = get_cfg()
        self.model = build_model(cfg).to(self.device).eval()
        cu.load_test_checkpoint(cfg, self.model)
        pdb.set_trace()


def crop_and_pad(frame, box, margin_percent):
    """Crop box with margin and take square crop from frame."""
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1

    # Add margin
    margin_x, margin_y = int(w * margin_percent / 100), int(h * margin_percent / 100)
    x1, y1 = max(0, x1 - margin_x), max(0, y1 - margin_y)
    x2, y2 = min(frame.shape[1], x2 + margin_x), min(frame.shape[0], y2 + margin_y)

    # Take square crop from frame
    size = max(y2 - y1, x2 - x1)
    center_y, center_x = (y1 + y2) // 2, (x1 + x2) // 2
    half_size = size // 2
    square_crop = frame[
        max(0, center_y - half_size) : min(frame.shape[0], center_y + half_size),
        max(0, center_x - half_size) : min(frame.shape[1], center_x + half_size),
    ]

    return cv2.resize(square_crop, (224, 224), interpolation=cv2.INTER_LINEAR)


def run(
    weights: str = "yolov8n.pt",
    device: str = "",
    source: str = "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    output_path: Optional[str] = None,
    crop_margin_percentage: int = 10,
    num_video_sequence_samples: int = 8,
    skip_frame: int = 2,
    video_cls_overlap_ratio: float = 0.25,
    fp16: bool = False,
    video_classifier_model: str = "microsoft/xclip-base-patch32",
    labels: List[str] = None,
) -> None:
    """
    Run action recognition on a video source using YOLO for object detection and a video classifier.

    Args:
        weights (str): Path to the YOLO model weights. Defaults to "yolov8n.pt".
        device (str): Device to run the model on. Use 'cuda' for NVIDIA GPU, 'mps' for Apple Silicon, or 'cpu'. Defaults to auto-detection.
        source (str): Path to mp4 video file or YouTube URL. Defaults to a sample YouTube video.
        output_path (Optional[str], optional): Path to save the output video. Defaults to None.
        crop_margin_percentage (int, optional): Percentage of margin to add around detected objects. Defaults to 10.
        num_video_sequence_samples (int, optional): Number of video frames to use for classification. Defaults to 8.
        skip_frame (int, optional): Number of frames to skip between detections. Defaults to 4.
        video_cls_overlap_ratio (float, optional): Overlap ratio between video sequences. Defaults to 0.25.
        fp16 (bool, optional): Whether to use half-precision floating point. Defaults to False.
        video_classifier_model (str, optional): Name or path of the video classifier model. Defaults to "microsoft/xclip-base-patch32".
        labels (List[str], optional): List of labels for zero-shot classification. Defaults to predefined list.

    Returns:
        None</edit>
    """
    if labels is None:
        labels = [
            "walking",
            "running",
            "brushing teeth",
            "looking into phone",
            "weight lifting",
            "cooking",
            "sitting",
        ]
    # Initialize models and device
    device = select_device(device)
    yolo_model = YOLO(weights).to(device)
    video_classifier = SlowfastVideoClassifier(video_classifier_model, device=device)
    pdb.set_trace()

    # Initialize video capture
    if source.startswith("http") and urlparse(source).hostname in {"www.youtube.com", "youtube.com", "youtu.be"}:
        source = get_best_youtube_url(source)
    elif not source.endswith(".mp4"):
        raise ValueError("Invalid source. Supported sources are YouTube URLs and MP4 files.")
    cap = cv2.VideoCapture(source)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize VideoWriter
    if output_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Initialize track history
    track_history = defaultdict(list)
    frame_counter = 0

    track_ids_to_infer = []
    crops_to_infer = []
    pred_labels = []
    pred_confs = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_counter += 1

        # Run YOLO tracking
        results = yolo_model.track(frame, persist=True, classes=[0])  # Track only person class

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()

            # Visualize prediction
            annotator = Annotator(frame, line_width=3, font_size=10, pil=False)

            if frame_counter % skip_frame == 0:
                crops_to_infer = []
                track_ids_to_infer = []

            for box, track_id in zip(boxes, track_ids):
                if frame_counter % skip_frame == 0:
                    crop = crop_and_pad(frame, box, crop_margin_percentage)
                    track_history[track_id].append(crop)

                if len(track_history[track_id]) > num_video_sequence_samples:
                    track_history[track_id].pop(0)

                if len(track_history[track_id]) == num_video_sequence_samples and frame_counter % skip_frame == 0:
                    start_time = time.time()
                    pdb.set_trace()
                    # crops = video_classifier.preprocess_crops_for_video_cls(track_history[track_id])
                    # end_time = time.time()
                    # preprocess_time = end_time - start_time
                    # print(f"video cls preprocess time: {preprocess_time:.4f} seconds")
                    # crops_to_infer.append(crops)
                    # track_ids_to_infer.append(track_id)




def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="ultralytics detector model path")
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
        default=["dancing", "singing a song"],
        help="labels for zero-shot video classification",
    )
    return parser.parse_args()


def main(opt):
    """Main function."""
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

