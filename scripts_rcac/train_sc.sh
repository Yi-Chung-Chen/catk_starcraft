#!/bin/bash
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------------------------------------------------------------
# StarCraft Motion — BC Pre-training (Gautschi cluster)
#
# Usage:
#   Interactive (after sinteractive -p ai --gpus-per-node=1 -A dinouye):
#     bash scripts_rcac/train_sc.sh
#
#   Batch:
#     sbatch scripts_rcac/train_sc.sbatch
# ---------------------------------------------------------------

MY_EXPERIMENT="sc_pre_bc"
MY_TASK_NAME=$MY_EXPERIMENT"-rcac"

# Use SLURM-allocated GPUs if available, else default to 1
NGPUS=${SLURM_GPUS_ON_NODE:-1}

if [ "$NGPUS" -gt 1 ]; then
  # Multi-GPU DDP
  torchrun \
    --rdzv_id ${SLURM_JOB_ID:-0} \
    --rdzv_backend c10d \
    --rdzv_endpoint ${MASTER_ADDR:-localhost}:${MASTER_PORT:-29500} \
    --nnodes ${SLURM_NNODES:-1} \
    --nproc_per_node $NGPUS \
    -m src.run \
    experiment=$MY_EXPERIMENT \
    trainer=ddp \
    task_name=$MY_TASK_NAME \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    data.test_batch_size=8 \
    model.model_config.use_aux_loss=false \
    model.model_config.decoder.num_concepts=0
else
  # Single-GPU
  python -m src.run \
    experiment=$MY_EXPERIMENT \
    task_name=$MY_TASK_NAME \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    data.test_batch_size=8 \
    trainer.max_epochs=10 \
    model.model_config.use_aux_loss=false \
    model.model_config.decoder.num_concepts=0
fi

echo "train_sc.sh done!"
