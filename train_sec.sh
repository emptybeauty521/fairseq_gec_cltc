#!/usr/bin/env bash

# --use-encoder-classification使用Token级标记任务
# --positive-label-weight错误Token损失的权重
# --use-sentence-copying使用句子级复制任务
DATA_BIN=./data/cltc/finetune/cged_all_bin
CUDA_VISIBLE_DEVICES=1 nohup python train.py $DATA_BIN \
--save-dir ./model/ft_lang8_all_cged_all_cltc7_7 \
--max-epoch 5 \
--batch-size 128 \
--max-tokens 14464 \
--update-freq 2 \
--train-subset train1_0 \
--valid-subset valid --skip-invalid-size-inputs-valid-test \
--pretrained-model ./model/ft_lang8_all_cltc7_7/checkpoint4.pt \
--arch transformer \
--optimizer adam --adam-betas '(0.9, 0.98)' \
--lr 0.00001 --min-lr 1e-09 --lr-scheduler inverse_sqrt \
--warmup-init-lr 1e-06 --warmup-updates 59 \
--dropout 0.2 --relu-dropout 0.2 --attention-dropout 0.2 --copy-attention-dropout 0.2 \
--decoder-layers 6 --encoder-layers 6 \
--encoder-embed-dim 512 --decoder-embed-dim 512 \
--max-target-positions 113 --max-source-positions 113 \
--encoder-ffn-embed-dim 2048 --decoder-ffn-embed-dim 2048 \
--encoder-attention-heads 8 --decoder-attention-heads 8 \
--share-all-embeddings \
--no-progress-bar \
--log-interval 10 \
--positive-label-weight 2.5 \
--copy-attention --copy-attention-heads 1 \
--no-ema \
--use-encoder-classification \
--seed 1 \
> log/ft_lang8_all_cged_all_cltc7_7.log &
