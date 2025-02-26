# conda activate yolov8

# DATA=Okutama-Synthetic-S2.yaml
# NAME=Detect_Syn_D1N

# python train.py --data $DATA --name $NAME




# DATA=Okutama-S2-D1N-v2-eval-D1D2N.yaml
# NAME=S2_D1N_v2_eval_D2N
# python train.py --data $DATA --name $NAME


DATA=Okutama-S2-D1N-v2-eval-D1D2N.yaml
NAME=S3_D1M_v1_eval_D1M
python train.py --data $DATA --name $NAME




# DATA=Okutama-D-D2Npart-D1D2N.yaml
# NAME=D_2.2.2 # Train 1.2.2 / Test D1D2Noon
# python train.py --data $DATA --name $NAME


# DATA=Okutama-S2-D1N-v2-eval-D1D2N.yaml
# NAME=S2_Drone1_Noon_v2-val_Drone2_Noon # Train 1.2.2 / Test D1D2Noon
# python train.py --data $DATA --name $NAME