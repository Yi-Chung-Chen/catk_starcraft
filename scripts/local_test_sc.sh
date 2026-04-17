#!/bin/bash
# Usage: bash scripts/local_test_sc.sh <ckpt> [variant] [dataset] [maps] [save_rollouts]
#   ckpt:           path to .ckpt file OR run directory (appends /checkpoints/last.ckpt)
#   variant:        smart (default), hmart, hmart_aux, hmart_c8, hmart_c32,
#                   hmart_intent, hmart_intent_aux, smart_intent_aux,
#                   hmart_c4_intent_aux, hmart_c8_intent_aux
#   dataset:        adv (default), unbias
#   maps:           id (default; 5 train/val maps: Abyssal_Reef_LE, Acolyte_LE,
#                       Ascension_to_Aiur_LE, Interloper_LE, Mech_Depot_LE),
#                   ood (held-out maps: Catallena_LE_(Void), Odyssey_LE),
#                   all (use config's test_map_names — null = every map in test split),
#                   custom comma list (e.g. "Catallena_LE_(Void),Odyssey_LE")
#   save_rollouts:  yes (default for test runs), no
#
# Example:
#   bash scripts/local_test_sc.sh logs/.../last.ckpt smart adv id  yes
#   bash scripts/local_test_sc.sh logs/.../last.ckpt smart adv ood yes

set -e

if [ -z "$1" ]; then
  echo "Error: missing checkpoint argument"
  echo "Usage: bash scripts/local_test_sc.sh <ckpt> [variant] [dataset] [maps] [save_rollouts]"
  exit 1
fi

CKPT_INPUT=$1
VARIANT=${2:-smart}
DATASET=${3:-adv}
MAPS=${4:-id}
SAVE_ROLLOUTS=${5:-yes}

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
# Test runs are save-and-go: no live iteration, no metric dashboarding.
# All numbers come from the offline harness (scripts/eval_sc_rollouts.py).
# Override with WANDB_MODE=online if you ever want the two test_closed/* scalars.
export WANDB_MODE=${WANDB_MODE:-disabled}

MY_EXPERIMENT="sc_pre_bc"
MY_TASK_NAME="sc_closed_test_${VARIANT}"

# --- Dataset ---
if [ "$DATASET" = "unbias" ]; then
  DATA_OVERRIDES="data=starcraft_unbias"
else
  DATA_OVERRIDES=""
fi

# --- Map filter ---
# Note on Catallena_LE_(Void): the map_name contains parentheses, which Hydra
# requires to be inside a quoted string element. Single-quote the element
# inside the bash double-quoted value.
case "$MAPS" in
  id)
    MAP_OVERRIDE='data.test_map_names=[Abyssal_Reef_LE,Acolyte_LE,Ascension_to_Aiur_LE,Interloper_LE,Mech_Depot_LE]'
    ;;
  ood)
    MAP_OVERRIDE="data.test_map_names=['Catallena_LE_(Void)',Odyssey_LE]"
    ;;
  all)
    MAP_OVERRIDE=""
    ;;
  *)
    # Custom comma-list. Hydra needs the list literal form. If a map name
    # contains parens or other special chars, quote it: "'Foo_(Bar)',Baz".
    MAP_OVERRIDE="data.test_map_names=[${MAPS}]"
    ;;
esac

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

SAVE_OVERRIDE=""
if [ "$SAVE_ROLLOUTS" = "yes" ]; then
  SAVE_OVERRIDE="model.model_config.save_closed_rollouts=true"
fi

if [ "$CONDA_DEFAULT_ENV" != "catk" ]; then
  echo "Error: catk conda env not active. Run 'conda activate catk' first."
  exit 1
fi

echo "=== local_test_sc.sh | variant=$VARIANT | dataset=$DATASET | maps=$MAPS | save=$SAVE_ROLLOUTS ==="
echo "    ckpt: $MY_CKPT_PATH"

# Closed-loop test on a single GPU. action=test runs trainer.test() which
# dispatches to SCSMART.test_step (aliased to _shared_eval_step(stage='test')).
# val_open_loop=false: skip token-cls; we only want the rollouts.
# val_closed_loop=true: enable closed-loop rollouts (the one we save).
python \
  -m src.run \
  action=test \
  experiment=$MY_EXPERIMENT \
  trainer=default \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.strategy=auto \
  ckpt_path=$MY_CKPT_PATH \
  model.model_config.val_closed_loop=true \
  model.model_config.val_open_loop=false \
  model.model_config.n_rollout_closed_val=1 \
  model.model_config.n_vis_batch=2 \
  model.model_config.n_vis_scenario=2 \
  task_name=$MY_TASK_NAME \
  $DATA_OVERRIDES \
  $MAP_OVERRIDE \
  $MODEL_OVERRIDES \
  $SAVE_OVERRIDE

echo "bash local_test_sc.sh done!"
