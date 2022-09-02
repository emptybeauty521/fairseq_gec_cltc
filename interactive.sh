#!/usr/bin/env bash

python interactive.py ./data/raw \
--path ./model/checkpoint_best.pt \
--beam 12 \
--nbest 1 \
--max-tokens 3400 \
--raw-text \
--batch-size 32 --buffer-size 32 \
--max-len-a 0 \
--max-len-b 122 \
--no-early-stop \
--copy-ext-dict \
--replace-unk \
--print-alignment \
--no-progress-bar
