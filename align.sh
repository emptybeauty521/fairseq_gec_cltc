#!/usr/bin/env bash

set -e
set -x

splits_num=1
threads_num=2
train_file="train"
RAW_DATA=./data/dae/pretrain_opt/ft_hq_norm

for ((split = 1; split <= $splits_num; split++))
do
    trainpref=$RAW_DATA/$train_file$split"_0"
    data_align=$RAW_DATA/align$split

    mkdir $data_align

    nohup python scripts/build_sym_alignment.py \
    --fast_align_dir fast_align/build/ \
    --mosesdecoder_dir mosesdecoder \
    --source_file $trainpref.src \
    --target_file $trainpref.tgt \
    --output_dir $data_align \
    > $RAW_DATA/align$split.log &

    if [ `expr $split % $threads_num` == 0 ]
    then
        wait
    fi
done

wait

for ((split = 1; split <= $splits_num; split++))
do
    data_align=$RAW_DATA/align$split
    trainpref=$RAW_DATA/$train_file$split"_0"
    cp $data_align/align.forward $trainpref.forward
    rm -rf $data_align
done
