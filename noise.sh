#!/usr/bin/env bash

set -e
set -x

txt_num=3072447
num_per_split=620000
splits_num=`expr $txt_num / $num_per_split`
if [ `expr $txt_num % $num_per_split` != 0 ]
then
    splits_num=`expr $splits_num + 1`
fi
threads_num=7

for ((split = 1; split <= $splits_num; split++))
do
    nohup python noise_data_ctc.py \
    --data arts_sents.txt \
    --split $split \
    --num_per_split $num_per_split \
    --seed 1 > data/ctc/pretrain/noise_data$split.log &

    if [ `expr $split % $threads_num` == 0 ]
    then
        wait
    fi
done
