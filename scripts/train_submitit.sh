#!/bin/bash

# Parse command line arguments
JOB_DIR=/checkpoint/amaia/video/raktimgg/HOWM/experiments/egodex_and_droid
ENV_FLAGS="LD_PRELOAD=/engshare/nccl/nccl-2.27.3/build/lib/libnccl.so.2.27.3"
DEBUG=0
QOS=explore
USE_FSDP=""
for arg in "$@"; do
  if [ "$arg" == "--debug" ]; then
    # DEBUG="--preview"
    DEBUG=1
    set -- "${@/--debug/}"
    continue
  fi
  if [ "$arg" == "--use_fsdp" ]; then
    USE_FSDP="--use_fsdp"
    set -- "${@/--use_fsdp/}"
    continue
  fi
  if [[ $arg == --qos=* ]]; then
    QOS="${arg#*=}"
    set -- "${@/--qos=*/}"
    continue
  fi
done

if [ "$DEBUG" -eq 1 ]; then
    export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    export MASTER_PORT=9901
    JOB_DIR=${JOB_DIR}/debug

    eval "$ENV_FLAGS torchrun --nnodes 1 --nproc-per-node 8 --rdzv-backend c10d --rdzv-endpoint $MASTER_ADDR:$MASTER_PORT \
    submitit_train_cw.py --nodes 1 --cpus_per_task 16 --timeout 1440 --partition learn --qos $QOS --config configs/egodex_and_droid.yaml --job_dir $JOB_DIR --debug $USE_FSDP"
else
    eval "$ENV_FLAGS python submitit_train_cw.py --nodes 32 --cpus_per_task 16 --timeout 1440 --partition learn --qos $QOS --config configs/egodex_and_droid.yaml --job_dir $JOB_DIR $USE_FSDP"
fi
