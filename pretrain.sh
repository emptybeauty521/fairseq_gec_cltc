#!/usr/bin/env bash


#--clip-norm 5 \
#以n个batch分别对深度神经网络做前向传播、计算损失、后向传播而计算网络参数的梯度，累积n个batch的梯度，最后更新参数；
#每个batch对参数的梯度都是独立计算的
#--use-sentence-copying \
#--use-encoder-classification \


#DATA_BIN=./data/bin/test
DATA_BIN=./data/cltc/pretrain/seed2
CUDA_VISIBLE_DEVICES=3 nohup python train.py $DATA_BIN \
--save-dir ./model/pretrain_seed2_cltc \
--max-epoch 10 \
--batch-size 128 \
--max-tokens 14464 \
--update-freq 2 \
--train-subset train1_0 \
--valid-subset valid --skip-invalid-size-inputs-valid-test \
--arch transformer \
--optimizer adam --adam-betas '(0.9, 0.98)' \
--lr 0.0002 --min-lr 1e-09 --lr-scheduler inverse_sqrt \
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
--positive-label-weight 3 \
--copy-attention --copy-attention-heads 1 \
--no-ema \
--use-encoder-classification \
--label-gen-loss-rate 1 \
--seed 2 \
> log/pretrain_seed2_cltc.log &
