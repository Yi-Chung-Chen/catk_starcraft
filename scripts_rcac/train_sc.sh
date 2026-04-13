#!/bin/bash
# Usage: bash scripts_rcac/train_sc.sh [variant] [dataset]
#   variant: smart (default), hmart, hmart_aux, hmart_c8, hmart_c32,
#            hmart_intent, hmart_intent_aux
#   dataset: adv (default), unbias

export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

VARIANT=${1:-smart}
DATASET=${2:-adv}

MY_EXPERIMENT="sc_pre_bc"
MY_TASK_NAME="$VARIANT-rcac"
BATCH_SIZE=16

# --- Dataset ---
if [ "$DATASET" = "unbias" ]; then
  DATA_OVERRIDES="data=starcraft_unbias"
else
  DATA_OVERRIDES=""
fi

# --- Variant-specific model flags ---
case "$VARIANT" in
  smart)
    MODEL_OVERRIDES="model.model_config.use_aux_loss=false model.model_config.decoder.num_concepts=0"
    ;;
  hmart)
    MODEL_OVERRIDES="model.model_config.use_aux_loss=false model.model_config.decoder.num_concepts=16"
    ;;
  hmart_aux)
    MODEL_OVERRIDES="model.model_config.use_aux_loss=true model.model_config.decoder.num_concepts=16"
    ;;
  hmart_c8)
    MODEL_OVERRIDES="model.model_config.use_aux_loss=false model.model_config.decoder.num_concepts=8"
    ;;
  hmart_c32)
    MODEL_OVERRIDES="model.model_config.use_aux_loss=false model.model_config.decoder.num_concepts=32"
    BATCH_SIZE=8
    ;;
  hmart_intent)
    MODEL_OVERRIDES="model.model_config.use_aux_loss=false model.model_config.decoder.num_concepts=16 model.model_config.decoder.use_action_target_input=true model.model_config.decoder.closed_loop_oracle_intent_input=true"
    ;;
  hmart_intent_aux)
    MODEL_OVERRIDES="model.model_config.use_aux_loss=true model.model_config.decoder.num_concepts=16 model.model_config.decoder.use_action_target_input=true"
    ;;
  *)
    echo "Unknown variant: $VARIANT"; exit 1
    ;;
esac

echo "=== train_sc.sh | variant=$VARIANT | dataset=$DATASET | batch=$BATCH_SIZE ==="

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
    data.train_batch_size=$BATCH_SIZE \
    data.val_batch_size=$BATCH_SIZE \
    data.test_batch_size=$BATCH_SIZE \
    trainer.max_epochs=10 \
    $DATA_OVERRIDES \
    $MODEL_OVERRIDES
else
  python -m src.run \
    experiment=$MY_EXPERIMENT \
    task_name=$MY_TASK_NAME \
    data.train_batch_size=$BATCH_SIZE \
    data.val_batch_size=$BATCH_SIZE \
    data.test_batch_size=$BATCH_SIZE \
    trainer.max_epochs=10 \
    $DATA_OVERRIDES \
    $MODEL_OVERRIDES
fi

echo "train_sc.sh done!"
