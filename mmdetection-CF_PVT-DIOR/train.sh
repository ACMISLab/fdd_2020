#!/usr/bin/env bash
CONFIG=$1
GPU_NUMS=$2
PORT=${PORT:-29506}
CUDA_VISIBLE_DEVICES="2,3" \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPU_NUMS --master_port=$PORT $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}