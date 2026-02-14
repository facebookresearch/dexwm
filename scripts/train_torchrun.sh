#!/usr/bin/env bash

JOB_DIR=""
if [[ $# -ge 2 && "$1" == "--job_dir" ]]; then
  JOB_DIR="$2"
elif [[ $# -ge 1 ]]; then
  JOB_DIR="$1"
else
  echo "Usage: $0 --job_dir <job_dir>  (or: $0 <job_dir>)" >&2
  exit 1
fi

export NUM_NODES=1
export HOST_NODE_ADDR=localhost
export CURR_NODE_RANK=0
export NUM_PROC_PER_NODES=8

torchrun \
  --nnodes=${NUM_NODES} \
  --nproc-per-node=${NUM_PROC_PER_NODES} \
  --node-rank=${CURR_NODE_RANK} \
  --rdzv-backend=c10d \
  --rdzv-endpoint=${HOST_NODE_ADDR}:29500 \
  train_wm.py --config configs/egodex_and_droid.yaml --job_dir "${JOB_DIR}"
