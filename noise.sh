#!/usr/bin/env bash

# 文章数量，每分片的文章数量，并行处理的进程数量
txt_num=3072447
num_per_split=620000
threads_num=7

# 分片数量
splits_num=`expr $txt_num / $num_per_split`
if [ `expr $txt_num % $num_per_split` != 0 ]
then
    splits_num=`expr $splits_num + 1`
fi

# 生成伪纠错数据
for ((split = 1; split <= $splits_num; split++))
do
    nohup python noise_data_cltc.py \
    --data arts_sents.txt \
    --split $split \
    --num_per_split $num_per_split \
    --seed 1 > data/cltc/pretrain/noise_data$split.log &

    if [ `expr $split % $threads_num` == 0 ]
    then
        wait
    fi
done
