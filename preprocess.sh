#!/usr/bin/env bash

set -e
set -x

RAW_DATA=./data/cltc/finetune
BIN_DATA=./data/cltc/finetune/bin

# set copy params
copy_params='--copy-ext-dict'

# set common params between train/valid
common_params='--source-lang src --target-lang tgt  
--padding-factor 1 
--srcdict ./data/cltc/pretrain/dict.src.txt
--joined-dictionary 
'

splits_num=1
threads_num=12
train_file="train"
valid_file="valid"

# preprocess train
for ((split = 1; split <= $splits_num; split++))
do
    trainpref=$RAW_DATA/$train_file$split"_0"

    nohup python preprocess.py \
    $common_params \
    $copy_params \
    --trainpref $trainpref \
    --destdir $BIN_DATA \
    --output-format binary \
    --alignfile $trainpref.forward \
    > $BIN_DATA/preprocess_$train_file$split.log &

    if [ `expr $split % $threads_num` == 0 ]
    then
        wait
    fi
done

# validpref=$RAW_DATA/$valid_file
#
# # preprocess valid, 不需要alignfile
# nohup python preprocess.py \
# $common_params \
# $copy_params \
# --validpref $validpref \
# --destdir $BIN_DATA \
# --output-format binary \
# > $BIN_DATA/preprocess_$valid_file.log &
