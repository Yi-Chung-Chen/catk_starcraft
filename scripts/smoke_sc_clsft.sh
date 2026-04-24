#!/bin/bash
# Usage: bash scripts/smoke_sc_clsft.sh <bc_ckpt_path> [variant] [batch_size]
#   bc_ckpt_path: REQUIRED, path to BC pre-trained checkpoint
#   variant: same set as train_sc_clsft.sh (default: hmart_intent_aux)
#   batch_size: train batch size (default: 16). val_batch_size stays at 4.
#
# Runs 5 train batches + 5 val batches, 1 epoch. Triggers the closed-loop val
# path so you can confirm the larger batch fits without committing to a full run.
# Uses task_name=<variant>_clsft_smoke so wandb/hydra runs don't collide with
# the real CLSFT run.

export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BC_CKPT=${1:?"BC checkpoint path required as first arg"}
VARIANT=${2:-hmart_intent_aux}
BATCH_SIZE=${3:-16}

if [ ! -f "$BC_CKPT" ]; then
  echo "BC checkpoint not found: $BC_CKPT"; exit 1
fi

MY_EXPERIMENT="sc_clsft"
MY_TASK_NAME="${VARIANT}_clsft_smoke"

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
    ;;
  hmart_intent)
    MODEL_OVERRIDES="model.model_config.use_aux_loss=false model.model_config.decoder.num_concepts=16 model.model_config.decoder.use_action_target_input=true model.model_config.decoder.closed_loop_oracle_intent_input=true"
    ;;
  hmart_intent_aux)
    MODEL_OVERRIDES="model.model_config.use_aux_loss=true model.model_config.decoder.num_concepts=16 model.model_config.decoder.use_action_target_input=true"
    ;;
  smart_intent_aux)
    MODEL_OVERRIDES="model.model_config.use_aux_loss=true model.model_config.decoder.num_concepts=0 model.model_config.decoder.use_action_target_input=true"
    ;;
  hmart_c4_intent_aux)
    MODEL_OVERRIDES="model.model_config.use_aux_loss=true model.model_config.decoder.num_concepts=4 model.model_config.decoder.use_action_target_input=true"
    ;;
  hmart_c8_intent_aux)
    MODEL_OVERRIDES="model.model_config.use_aux_loss=true model.model_config.decoder.num_concepts=8 model.model_config.decoder.use_action_target_input=true"
    ;;
  *)
    echo "Unknown variant: $VARIANT"; exit 1
    ;;
esac

echo "=== smoke_sc_clsft.sh | variant=$VARIANT | batch_size=$BATCH_SIZE | bc_ckpt=$BC_CKPT ==="

torchrun --nproc_per_node=gpu \
  -m src.run \
  experiment=$MY_EXPERIMENT \
  task_name=$MY_TASK_NAME \
  $MODEL_OVERRIDES \
  ckpt_path=$BC_CKPT \
  data.train_batch_size=$BATCH_SIZE \
  trainer.limit_train_batches=5 \
  trainer.limit_val_batches=5 \
  trainer.max_epochs=1

echo "smoke_sc_clsft.sh done!"
