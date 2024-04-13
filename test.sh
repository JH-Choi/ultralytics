# MODEL=runs_yolov8n/train_15_40_val_50_40_real_only/train/weights/best.pt
# MODEL=runs_yolov8n/train_15_40_val_50_40_synthetic/train/weights/best.pt
# MODEL=runs_yolov8n/train_15_40_val_50_40_combined/train/weights/best.pt

# Input: Video
# MODEL=ckpts/yolov8n.pt
# yolo task=detect mode=predict \
#   model=$MODEL \
#   show=True conf=0.5 \
#   source=/mnt/hdd/data/Archangel/Archangel_50m/AA_BP_12_45_50_40_20211123_1637682382_EO.mp4


# Input: Images
# MODEL=ckpts/yolov8n.pt
# yolo task=detect mode=predict \
#   model=$MODEL \
#   show=True conf=0.5 \
#   source=/mnt/hdd/data/Okutama_Action/TestSetFrames/Drone1/Morning/Extracted-Frames-1280x720/1.1.8

