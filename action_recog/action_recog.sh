
# Preprocess for action recognition

DATA_PATH=/mnt/hdd/data/Okutama_Action/ActionRecognition/TrainSetFrames

# SYN_PATH=/mnt/hdd/code/human_data_generation/xrfeitoria/output/S2_orig_Drone1_Noon_1_2_2/auto_Drone1_Noon_1_2_2_alti0/composite_w_shadow/actions
# SYN_DATA_PATH=$DATA_PATH/syn_1.2.2
# ln -s $SYN_PATH $SYN_DATA_PATH

# TRAIN_IDXS=('syn_1.2.2' 'syn_1.2.2/1.2.2' 'syn_1.2.2/2.2.2' '1.1.1' '2.1.1' '2.2.2' '1.2.2')
TRAIN_IDXS=('syn_1.2.2/2.1.1' 'syn_1.2.2/1.1.1')
# TRAIN_IDXS=('1.1.1' '2.1.1' '2.2.2')
TEST_IDXS='1.2.1/1.2.3/1.2.10/2.2.1/2.2.3/2.2.10'
# TEST_IDXS='1.1.8/1.1.9/2.1.8/2.1.9'
OUT_PATH=./output/swn3d_t
# OUT_PATH=./output/r3d_18

for TRAIN_IDX in ${TRAIN_IDXS[@]}; do
    # OUT_FOLDER=$OUT_PATH/real_$TRAIN_IDX
    OUT_FOLDER=$OUT_PATH/$TRAIN_IDX
    python train_real.py --train-idxs $TRAIN_IDX --test-idxs $TEST_IDXS --output-path $OUT_FOLDER
done




