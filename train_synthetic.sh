# conda activate yolov8

# DATA=Okutama-Synthetic-S2.yaml
# NAME=Detect_Syn_D1N

# python train.py --data $DATA --name $NAME




# DATA=Okutama-S2-D1N-v2-eval-D1D2N.yaml
# NAME=S2_D1N_v2_eval_D2N
# python train.py --data $DATA --name $NAME


# MODEL_NAME=yolov8n # yolov8n / yolov8s / yolov8m 

for MODEL_NAME in yolov8n yolov8s yolov8m
do
    DATA=Okutama-S2-v1-eval-D1D2N.yaml
    NAME=$MODEL_NAME/S2_v1_eval_D1D2N
    python train.py --data $DATA --name $NAME --model_name $MODEL_NAME

    DATA=Okutama-S2-v2-eval-D1D2N.yaml
    NAME=$MODEL_NAME/S2_v2_eval_D1D2N
    python train.py --data $DATA --name $NAME --model_name $MODEL_NAME

    DATA=Okutama-S3-v1-eval-D1D2M.yaml
    NAME=$MODEL_NAME/S3_v1_eval_D1D2M
    python train.py --data $DATA --name $NAME --model_name $MODEL_NAME 

    DATA=Okutama-S3-v2-eval-D1D2M.yaml
    NAME=$MODEL_NAME/S3_v2_eval_D1D2M
    python train.py --data $DATA --name $NAME --model_name $MODEL_NAME  
done




# DATA=Okutama-D-D1Mpart-D1D2M.yaml
# DATA=Okutama-D-D2Npart-D1D2N.yaml
# MODEL_NAME=yolov8s # yolov8n / yolov8s / yolov8m 
# NAME=D_1.1.1 # Train 1.2.2 / Test D1D2Noon
# NAME=$MODEL_NAME/D_1.1.1 # Train 1.2.2 / Test D1D2Noon
# NAME=$MODEL_NAME/D_2.2.1 # Train 1.2.2 / Test D1D2Noon
# NAME=$MODEL_NAME/D_2.2.4 # Train 1.2.2 / Test D1D2Noon
# NAME=yolov8s/D_2.2.2 # Train 1.2.2 / Test D1D2Noon
# python train.py --data $DATA --name $NAME



# DATA=Okutama-D-D2Npart-D1D2N.yaml
# NAME=D_2.2.2 # Train 1.2.2 / Test D1D2Noon
# python train.py --data $DATA --name $NAME


# DATA=Okutama-S2-D1N-v2-eval-D1D2N.yaml
# NAME=S2_Drone1_Noon_v2-val_Drone2_Noon # Train 1.2.2 / Test D1D2Noon
# python train.py --data $DATA --name $NAME