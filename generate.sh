#!/usr/bin/env bash

set -x
set -e

#generate.py中都为True而都启用
#--copy-ext-dict \
#启用align_dict
#--replace-unk \    ?
#利用对齐字典/信息
#--print-alignment \
#计算对齐信息
#--cpu \

#--no-progress-bar \
#--no-early-stop \
#--lenpen \
#--unkpen \
#去掉评分、概率、src、tgt，bleu评估

TEST_LABLE="test_pm"
DATA_PATH=./data/benchmark

CUDA_VISIBLE_DEVICES=0 nohup python generate.py $DATA_PATH \
--path ./model/ft_sec_cp_69k/checkpoint_best.pt:./model/ft_sec_pos_neg/checkpoint_best.pt \
--beam 1 \
--nbest 1 \
--gen-subset bm \
--max-tokens 31232 \
--raw-text \
--batch-size 256 \
--max-len-a 0 \
--max-len-b 122 \
--cpu \
> $DATA_PATH/predict_log_$TEST_LABLE.txt &

wait

cat $DATA_PATH/predict_log_$TEST_LABLE.txt | grep "^H" | python ./gec_scripts/sort.py 1 $DATA_PATH/predict_result_$TEST_LABLE.txt
