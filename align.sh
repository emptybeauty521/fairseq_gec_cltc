#!/usr/bin/env bash

RAW_DATA=./data/cltc/finetune

# 分片数量，并行处理的进程数量
splits_num=1
threads_num=2

train_file="train"

# 多进程生成训练数据的对齐信息
# 文件前缀为：train{n}_0，其中n为1、2、3...n
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

# 复制生成的对齐信息
for ((split = 1; split <= $splits_num; split++))
do
    data_align=$RAW_DATA/align$split
    trainpref=$RAW_DATA/$train_file$split"_0"
    cp $data_align/align.forward $trainpref.forward
    rm -rf $data_align
done
