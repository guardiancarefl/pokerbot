#!/bin/bash
cd /home/quant/pokerbot
source .venv/bin/activate
export PYTHONUNBUFFERED=1
# Truncate log fresh, then APPEND-only — both writers atomic
: > /tmp/k200_real_ante_train.log
exec python -u scripts/continue_k200_real_ante.py \
    --resume runs/k200_real_ante_20260605_225847/ckpt_iter_0100.pt \
    --target-iter 2000 \
    --checkpoint-every 100 \
    >> /tmp/k200_real_ante_train.log 2>&1
