#!/bin/bash

#SBATCH --job-name=fine_tune_robocasa_v27_dino_small_clip_train
#SBATCH --output=/checkpoint/siro/xavierpuig/vla/logs/%A/log_%a.log
#SBATCH --error=/checkpoint/siro/xavierpuig/vla/logs/%A/log_%a.err
#SBATCH --time=10080
#SBATCH --nodes 1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=90
#SBATCH --mem=1900G
#SBATCH --account=siro
#SBATCH --qos=h200_siro_high
#SBATCH --array=0-0

# Print out the job related IDs
echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID
echo "SLURM_JOB_NUM_NODES: " $SLURM_JOB_NUM_NODES

# This is needed to run multi-nodes training
export NCCL_NSOCKS_PERTHREAD=4 # specifies the number of sockets opened by each helper thread of the socket transport
export NCCL_SOCKET_NTHREADS=2 # specifies the number of CPU helper threads used per network connection for socket transport
export FI_EFA_SET_CUDA_SYNC_MEMOPS=0 # based on https://fb.workplace.com/groups/aws.fair.discuss/posts/1494441921474154/?comment_id=1494443018140711&reply_comment_id=1494449228140090

# GPU check
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
NUM_GPU="$(nvidia-smi --list-gpus | wc -l)"
echo "NUM_GPU=$NUM_GPU"

# Get the master address
NODES=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
NODE_ARRAY=($NODES)
HEAD_NODE=${NODE_ARRAY[0]}
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address)
JOBNAME="v27_dino_small_clip"


# Cd to robot-skills
cd /home/xavierpuig/code/projects/eval_roboskills_sim/robot-skills-sim/robocasa-murp/robocasa/scripts

# Unset LD_LIBRARY_PATH otherwise AWS cluster has an issue of loading pytorch
# source ~/miniconda/bin/activate
# conda activate vla
unset LD_LIBRARY_PATH
export TRANSFORMERS_CACHE=//checkpoint/siro/xavierpuig/weights/
export VLA_DATA_DIR=/checkpoint/siro/jimmytyyang/robot-skills-dataset
export VLA_LOG_DIR=/checkpoint/siro/xavierpuig/logs
export VLA_WANDB_ENTITY=xavierpuig
export WANDB_MODE=online

ckpt_list=(
    # "/checkpoint/siro/jtruong/repos/robot-skills/results/vla_small_egodex_sim_real_pap_224_torso_wrist_ee_6d_cmd_rgb_h50_dino_freezeVT_noImgAug_bf32_bs8_long/train_lr_5e-05_seed_42/checkpoint/step65000_0.081.pt"
    # "/checkpoint/siro/jtruong/repos/robot-skills/results/vla_small_egodex_real_pap_224_torso_wrist_ee_6d_cmd_rgb_h50_dino_freezeVT_noImgAug_bf32_bs8_long/train_lr_5e-05_seed_42/checkpoint/step270000_0.078.pt"
    # "/checkpoint/siro/jtruong/repos/robot-skills/results/vla_small_egodex_sim_pap_224_torso_wrist_ee_6d_cmd_rgb_h50_dino_freezeVT_noImgAug_bf32_bs8_long/train_lr_5e-05_seed_42/checkpoint/step70000_0.070.pt"
    # "/checkpoint/siro/jtruong/repos/robot-skills/results/vla_small_egodex_sim_pap_224_torso_wrist_ee_6d_cmd_rgb_h50_dino_freezeVT_noImgAug_bf32_bs8_long/train_lr_5e-05_seed_42/checkpoint/step125000_0.073.pt"
    # "/checkpoint/siro/xavierpuig/projects/vla_training/vla_robot_skills/robot_skills_pr/robot-skills/multirun/ckpt_jimmy/fine_tune_robocasa_v27_dino_small_human_murp_6_step35000_0.116.pt"
    "/checkpoint/siro/akshararai/robot-skills/vla_checkpoints/checkpoint/siro/akshararai/vla_wb_log/train/egodex_sim_hl3_hor50_frameTrue_act1e-4/2025-08-27_23-18_42/checkpoint/step30000_nan.pt"
)

export ckpt_name=${ckpt_list[$SLURM_ARRAY_TASK_ID]}
# export full_ckpt_name=/checkpoint/siro/xavierpuig/projects/vla_training/vla_robot_skills/robot_skills_pr/robot-skills/multirun/"${ckpt_name}"/checkpoint/*
echo "Running evaluation for checkpoint: ${full_ckpt_name}"



/home/xavierpuig/miniconda3/envs/robot-skills-sim/bin/python eval_vla.py \
    --vla_path="${ckpt_name}" \
    --dataset=/checkpoint/siro/jimmytyyang/robot-skills-sim-gen-large-scale/0801-episode-clutter-object-combined_demo_17.hdf5 \
    --max_steps=1500 \
    --video_path=/checkpoint/siro/xavierpuig/videos/final_res_verify_metrics_good/ \
    --use_abs_actions \
    --n 50 \
    --target_path /home/xavierpuig/code/projects/eval_roboskills_sim/robot-skills-sim/robocasa-murp/ \
    --downsample_factor 4 \
    --num_proces 8 \
    --num_gpu 4 \
    --use_abs_actions \
    --src_path storage/home/jimmytyyang/research/robot-skills-sim/robocasa-murp \
    --backup_dataset_path=/checkpoint/siro/jimmytyyang/robot-skills-sim-gen-large-scale/0801-episode-clutter-object-combined_demo_0.hdf5 \
    --latest_or_lowest_loss_checkpoint latest \
    --dynamic_simulation




