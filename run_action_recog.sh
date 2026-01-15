############################################################################################################
# Run examples/YOLOv8-Action-Recognition/action_recognition.py
############################################################################################################
# conda activate yolov8

# ### source video
# SRC_VIDEO=/mnt/hdd/data/Okutama_Action/TestSetVideos/Drone1/Morning/1.1.8.mp4
SRC_VIDEO=/mnt/hdd/data/Okutama_Action/TestSetVideos/Drone1/Noon/1.2.1.mp4
# # SRC_VIDEO=/mnt/hdd/code/ARLproject/ultralytics/sequence_1.mp4
# # SRC_VIDEO=/mnt/hdd/code/Action_Recognition/LART/assets/jump.mp4

# ### Model weights
MODEL=yolov8l.pt 
# MODEL=runs/detect/Detect_D1M_val_D1M/weights/best.pt

num_video_seq=8 # 8 (microsoft) or 16 (torchvision)
video_classifier_model=microsoft/xclip-base-patch32 # microsoft/xclip-base-patch32 / s3d / mvit_v2_s
skip_frame=1

python examples/YOLOv8-Action-Recognition/action_recognition.py --source $SRC_VIDEO --skip-frame $skip_frame \
--labels standing sitting walking running lying jumping --weights $MODEL --num-video-sequence-samples $num_video_seq --video-classifier-model $video_classifier_model



############################################################################################################
# Run action_recog/yolov8_pytorchvideo.py
############################################################################################################

# ### source video
# SRC_VIDEO=/mnt/hdd/data/Okutama_Action/TestSetVideos/Drone1/Morning/1.1.8.mp4
# # SRC_VIDEO=/mnt/hdd/code/ARLproject/ultralytics/sequence_1.mp4
# # SRC_VIDEO=/mnt/hdd/code/Action_Recognition/LART/assets/jump.mp4

# ### Model weights
# # MODEL=yolov8n.pt 
# MODEL=runs/detect/Detect_D1_Morn_val_D1_Morn/weights/best.pt

# num_video_seq=16 # 8 (microsoft) or 16 (torchvision)
# video_classifier_model=i3d_r50 
# skip_frame=1

# python action_recog/yolov8_pytorchvideo.py --source $SRC_VIDEO --skip-frame $skip_frame \
# --labels standing sitting walking running lying jumping --weights $MODEL --num-video-sequence-samples $num_video_seq --video-classifier-model $video_classifier_model

############################################################################################################
# Train 
############################################################################################################
# MODEL=yolov11n.pt 
# num_video_seq=16 # 8 (microsoft) or 16 (torchvision)
# video_classifier_model=i3d_r50 
# skip_frame=1

# python action_recog/train_video.py --source $SRC_VIDEO --skip-frame $skip_frame \
# --labels standing sitting walking running lying jumping --weights $MODEL --num-video-sequence-samples $num_video_seq --video-classifier-model $video_classifier_model



############################################################################################################
# Run action_recog/yolov8_slowfast.py
############################################################################################################

# ### source video
# SRC_VIDEO=/mnt/hdd/data/Okutama_Action/TestSetVideos/Drone1/Morning/1.1.8.mp4
# # SRC_VIDEO=/mnt/hdd/code/ARLproject/ultralytics/sequence_1.mp4
# # SRC_VIDEO=/mnt/hdd/code/Action_Recognition/LART/assets/jump.mp4

# ### Model weights
# # MODEL=yolov8n.pt 
# MODEL=runs/detect/Detect_D1_Morn_val_D1_Morn/weights/best.pt

# num_video_seq=16 # 8 (microsoft) or 16 (torchvision)
# video_classifier_model=i3d_r50 
# skip_frame=1

# python action_recog/yolov8_slowfast.py --source $SRC_VIDEO --skip-frame $skip_frame \
# --labels standing sitting walking running lying jumping --weights $MODEL --num-video-sequence-samples $num_video_seq --video-classifier-model $video_classifier_model


