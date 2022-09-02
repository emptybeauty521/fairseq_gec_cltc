#!/usr/bin/env bash

RAW_DATA=./data/cltc/pretrain
BIN_DATA=./data/cltc/pretrain/bin

copy_params='--copy-ext-dict'

common_params='--source-lang src --target-lang tgt
--padding-factor 1 
--srcdict ./data/cltc/pretrain/dict.src.txt
--joined-dictionary 
'

# 分片数量，并行处理的进程数量
splits_num=10
threads_num=12

train_file="train"
valid_file="valid"

# 多进程预处理训练数据
# 文件前缀为：train{n}_0，其中n为1、2、3...n
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

# 预处理验证数据，可不用对齐信息
validpref=$RAW_DATA/$valid_file
nohup python preprocess.py \
$common_params \
$copy_params \
--validpref $validpref \
--destdir $BIN_DATA \
--output-format binary \
> $BIN_DATA/preprocess_$valid_file.log &
