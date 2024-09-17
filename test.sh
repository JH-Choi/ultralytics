# MODEL=runs_yolov8n/train_15_40_val_50_40_real_only/train/weights/best.pt
# MODEL=runs_yolov8n/train_15_40_val_50_40_synthetic/train/weights/best.pt
# MODEL=runs_yolov8n/train_15_40_val_50_40_combined/train/weights/best.pt

### Input: Video
# MODEL=yolov8s.pt
# MODEL=runs/detect/Detect_D1_Morn_val_D1_Morn/weights/best.pt

# SRC_VIDEO=/mnt/hdd/data/Okutama_Action/TestSetVideos/Drone1/Morning/1.1.8.mp4
# # SRC_VIDEO=/mnt/hdd/data/Archangel/Archangel_35m/AA_BP_8_45_35_25_20211123_1637678962_EO.mp4
# yolo task=detect mode=predict \
#   model=$MODEL \
#   show=True conf=0.5 \
#   source=$SRC_VIDEO


### Input: Images
# MODEL=ckpts/yolov8n.pt
# yolo task=detect mode=predict \
#   model=$MODEL \
#   show=True conf=0.5 \
#   source=/mnt/hdd/data/Okutama_Action/TestSetFrames/Drone1/Morning/Extracted-Frames-1280x720/1.1.8


### Input: Images / Evaluation
# MODEL=runs/detect/Detect_D2_Morn_val_D2_Morn/weights/best.pt
# DATA_CFG=Okutama-Detect.yaml
# yolo task=detect mode=val \
#   model=$MODEL data=$DATA_CFG \
#   show=True 


### Input: Images / Evaluating multiple models
DATA=(
    "Okutama-D-D1M-D1M.yaml" 
    "Okutama-D-D2M-D2M.yaml" 
    "Okutama-D-D1N-D1N.yaml" 
    "Okutama-D-D2N-D2N.yaml" 
    "Okutama-D-D1D2M-D1D2M.yaml" 
    "Okutama-D-D1D2N-D1D2N.yaml" 
    "Okutama-D-ALL-ALL.yaml" 
) 
MODEL=(
    "runs/detect/Detect_D1M_val_D1M/weights/best.pt" 
    "runs/detect/Detect_D2M_val_D2M/weights/best.pt" 
    "runs/detect/Detect_D1N_val_D1N/weights/best.pt" 
    "runs/detect/Detect_D2N_val_D2N/weights/best.pt" 
    "runs/detect/Detect_D1D2M_val_D1D2M/weights/best.pt" 
    "runs/detect/Detect_D1D2N_val_D1D2N/weights/best.pt" 
    "runs/detect/Detect_ALL_val_ALL/weights/best.pt" 
)
NAME=(
    "Eval_D1M_D1M"
    "Eval_D2M_D2M"
    "Eval_D1N_D1N"
    "Eval_D2N_D2N"
    "Eval_D1D2M_D1D2M"
    "Eval_D1D2N_D1D2N"
    "Eval_ALL_ALL"
)



length=${#DATA[@]}

for (( i=0; i<$length; i++ ))
do
   echo "DATA ${DATA[$i]} | MODEL ${MODEL[$i]} | NAME ${NAME[$i]}"
   yolo task=detect mode=val \
     model="${MODEL[$i]}" data="${DATA[$i]}" name="${NAME[$i]}" \
     show=True
done

