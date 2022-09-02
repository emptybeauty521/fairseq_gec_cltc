#!/usr/bin/env bash

# --use-encoder-classification使用Token级标记任务
# --positive-label-weight错误Token损失的权重
DATA_BIN=./data/cltc/pretrain/bin
CUDA_VISIBLE_DEVICES=3 nohup python train.py $DATA_BIN \
--save-dir ./model/pretrain_cltc \
--max-epoch 10 \
--batch-size 128 \
--max-tokens 14464 \
--update-freq 2 \
--train-subset train1_0 \
--valid-subset valid --skip-invalid-size-inputs-valid-test \
--arch transformer \
--optimizer adam --adam-betas '(0.9, 0.98)' \
--lr 0.0003 --min-lr 1e-09 --lr-scheduler inverse_sqrt \
--warmup-init-lr 1e-07 --warmup-updates 16000 \
--save-interval-updates 19527 \
--dropout 0.2 --relu-dropout 0.2 --attention-dropout 0.2 --copy-attention-dropout 0.2 \
--decoder-layers 6 --encoder-layers 6 \
--encoder-embed-dim 512 --decoder-embed-dim 512 \
--max-target-positions 113 --max-source-positions 113 \
--encoder-ffn-embed-dim 2048 --decoder-ffn-embed-dim 2048 \
--encoder-attention-heads 8 --decoder-attention-heads 8 \
--share-all-embeddings \
--no-progress-bar \
--log-interval 20 \
--positive-label-weight 2.5 \
--copy-attention --copy-attention-heads 1 \
--no-ema \
--use-encoder-classification \
--seed 1 \
> log/pretrain_cltc.log &
