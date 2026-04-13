#!/bin/bash
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Ablation: hmart_c32 (global concepts=32, no aux)

MY_EXPERIMENT="sc_pre_bc"
MY_TASK_NAME="hmart_c32-rcac"

NGPUS=${SLURM_GPUS_ON_NODE:-1}

if [ "$NGPUS" -gt 1 ]; then
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
    trainer.max_epochs=10 \
    model.model_config.use_aux_loss=false \
    model.model_config.decoder.num_concepts=32
else
  python -m src.run \
    experiment=$MY_EXPERIMENT \
    task_name=$MY_TASK_NAME \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    data.test_batch_size=8 \
    trainer.max_epochs=10 \
    model.model_config.use_aux_loss=false \
    model.model_config.decoder.num_concepts=32
fi

echo "train_hmart_c32.sh done!"
