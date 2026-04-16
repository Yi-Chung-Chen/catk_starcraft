#!/bin/bash
# Usage: bash scripts/local_val_sc.sh <ckpt> [variant] [dataset] [observer]
#   ckpt:     path to .ckpt file OR run directory (appends /checkpoints/last.ckpt)
#   variant:  smart (default), hmart, hmart_aux, hmart_c8, hmart_c32,
#             hmart_intent, hmart_intent_aux, smart_intent_aux,
#             hmart_c4_intent_aux, hmart_c8_intent_aux
#   dataset:  adv (default), unbias
#   observer: both (default), p1, p2 — controls which observer's GIFs are written
#
# Example:
#   bash scripts/local_val_sc.sh logs/smart_intent_aux/runs/2026-04-16_03-16-47 smart_intent_aux

set -e

if [ -z "$1" ]; then
  echo "Error: missing checkpoint argument"
  echo "Usage: bash scripts/local_val_sc.sh <ckpt> [variant] [dataset] [observer]"
  exit 1
fi

CKPT_INPUT=$1
VARIANT=${2:-smart}
DATASET=${3:-adv}
OBSERVER=${4:-both}

# --- Resolve checkpoint path (accept either a directory or a .ckpt file) ---
if [ -d "$CKPT_INPUT" ]; then
  MY_CKPT_PATH="$CKPT_INPUT/checkpoints/last.ckpt"
elif [ -f "$CKPT_INPUT" ]; then
  MY_CKPT_PATH="$CKPT_INPUT"
else
  echo "Error: checkpoint not found: $CKPT_INPUT"
  exit 1
fi

if [ ! -f "$MY_CKPT_PATH" ]; then
  echo "Error: resolved checkpoint file does not exist: $MY_CKPT_PATH"
  exit 1
fi

export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2

MY_EXPERIMENT="sc_pre_bc"
MY_TASK_NAME="sc_closed_val_${VARIANT}"

# --- Dataset ---
if [ "$DATASET" = "unbias" ]; then
  DATA_OVERRIDES="data=starcraft_unbias"
else
  DATA_OVERRIDES=""
fi

# --- Variant-specific model flags (must match train_sc.sh to load the ckpt) ---
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

if [ "$CONDA_DEFAULT_ENV" != "catk" ]; then
  echo "Error: catk conda env not active. Run 'conda activate catk' first."
  exit 1
fi

echo "=== local_val_sc.sh | variant=$VARIANT | dataset=$DATASET | observer=$OBSERVER ==="
echo "    ckpt: $MY_CKPT_PATH"

# Closed-loop validation on a single GPU.
# limit_val_batches stays at 125 so the scalar metric is comparable across runs;
# n_vis_batch / n_vis_scenario are low because GIFs are for spot-checking.
python \
  -m src.run \
  action=validate \
  experiment=$MY_EXPERIMENT \
  trainer=default \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.strategy=auto \
  ckpt_path=$MY_CKPT_PATH \
  model.model_config.val_closed_loop=true \
  model.model_config.val_open_loop=false \
  model.model_config.n_rollout_closed_val=1 \
  model.model_config.n_vis_batch=5 \
  model.model_config.n_vis_scenario=2 \
  model.model_config.vis_observer_player=$OBSERVER \
  trainer.limit_val_batches=125 \
  task_name=$MY_TASK_NAME \
  $DATA_OVERRIDES \
  $MODEL_OVERRIDES

echo "bash local_val_sc.sh done!"
