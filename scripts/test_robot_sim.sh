#!/bin/bash
TASK=reach # place grasp reach
ENCODER=dinov2 # dinov2
NODES=1 # number of nodes to use for distributed testing
DATA_DIR=/checkpoint/amaia/video/raktimgg/transfer/split_files/split_files/pick-and-place-2.0

# Parse command line arguments
ENV_FLAGS="LD_PRELOAD=/engshare/nccl/nccl-2.27.3/build/lib/libnccl.so.2.27.3 NCCL_RAS_ENABLE=0 LD_LIBRARY_PATH=/engshare/libs/libEGL:$LD_LIBRARY_PATH"
USE_SUBMITIT=0
QOS=explore
TIMEOUT=300

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)
      TASK="$2"
      shift 2
      ;;
    --encoder)
      ENCODER="$2"
      shift 2
      ;;
    --kp_loss_weight)
      KP_LOSS_WEIGHT="$2"
      shift 2
      ;;
    --qos)
      QOS="$2"
      shift 2
      ;;
    --nodes)
      NODES="$2"
      shift 2
      ;;
    --timeout)
      TIMEOUT="$2"
      shift 2
      ;;
    --debug)
      USE_SUBMITIT=1
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      shift
      ;;
  esac
done

# Set encoder-specific variables
if [ "$ENCODER" == "dinov2" ]; then
  BASE_DIR=/checkpoint/amaia/video/raktimgg/HOWM/experiments/robocasa_random_finetune_var_time_multistep_kp_bug_corr_1e-5/
  CKPT_NAME=robocasa_random_heatmap_loss_17.pth.tar
  CONFIG=configs/robocasa_random_multistep.yaml
  DEC_CKPT=/checkpoint/amaia/video/davidfan/experiments/noco/decoder/vit_l_dinov2_vitl_nonorm_res224_egodex_4node/checkpoints/checkpoint_0055000/checkpoint.pt
  KP_LOSS_WEIGHT=${KP_LOSS_WEIGHT:-1e-3}
fi

# Update paths after encoder is set
CKPT_PATH=$BASE_DIR/checkpoints/$CKPT_NAME
JOB_DIR=$BASE_DIR/eval_test/robocasa/${TASK}_${KP_LOSS_WEIGHT}
# JOB_DIR=debug3

# Set task-specific flags
TASK_FLAGS=""
if [ "$TASK" == "grasp" ]; then
  TASK_FLAGS="--use_loss_ee"
fi

echo $JOB_DIR
echo $TASK
echo $ENCODER
echo $TASK_FLAGS
echo $KP_LOSS_WEIGHT
echo "Nodes: $NODES"

# Initialize conda for bash shell
eval "$(conda shell.bash hook)"
conda activate robot-skills-sim

if [ "$USE_SUBMITIT" -eq 0 ]; then
  eval "$ENV_FLAGS torchrun --nproc-per-node 8 --nnodes 1 \
    -m sim_test \
    --job_dir $JOB_DIR \
    --config $CONFIG \
    --checkpoint $CKPT_PATH \
    --method dexwm \
    --task_name $TASK \
    --kp_loss_weight $KP_LOSS_WEIGHT \
    --data_dir $DATA_DIR \
    --decoder_checkpoint $DEC_CKPT \
    $TASK_FLAGS"
else
  eval "$ENV_FLAGS python -m submit_test_sim_task --nodes $NODES --cpus_per_task 16 --timeout $TIMEOUT --partition learn --qos $QOS \
        --job_dir $JOB_DIR \
        --config $CONFIG \
        --checkpoint $CKPT_PATH \
        --method dexwm \
        --task_name $TASK \
        --kp_loss_weight $KP_LOSS_WEIGHT \
        --data_dir $DATA_DIR \
        --decoder_checkpoint $DEC_CKPT \
        $TASK_FLAGS"
fi
