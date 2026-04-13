#!/bin/sh
# Usage: bash scripts/train_sc_resume.sh [dataset]
#   dataset: adv (default), unbias

export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATASET=${1:-adv}

if [ "$DATASET" = "unbias" ]; then
  DATA_OVERRIDES="data=starcraft_unbias"
else
  DATA_OVERRIDES=""
fi

MY_EXPERIMENT="sc_pre_bc"
MY_TASK_NAME=$MY_EXPERIMENT"-debug"

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate catk

# Single-GPU
torchrun --nproc_per_node=gpu \
  -m src.run \
  experiment=$MY_EXPERIMENT \
  task_name=$MY_TASK_NAME \
  data.train_batch_size=8 \
  data.val_batch_size=8 \
  data.test_batch_size=8 \
  trainer.max_epochs=10 \
  ckpt_path=logs/sc_pre_bc-debug/runs/2026-03-31_16-46-54/checkpoints/last.ckpt \
  $DATA_OVERRIDES

# Multi-GPU DDP (uncomment for multi-GPU or multi-node):
# torchrun \
#   --rdzv_id $SLURM_JOB_ID \
#   --rdzv_backend c10d \
#   --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
#   --nnodes $NUM_NODES \
#   --nproc_per_node gpu \
#   -m src.run \
#   experiment=$MY_EXPERIMENT \
#   trainer=ddp \
#   task_name=$MY_TASK_NAME

echo "bash train_sc.sh done!"
