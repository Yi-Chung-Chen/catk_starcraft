#!/bin/bash
# Usage: sbatch scripts_rcac/run.sbatch eval_val_sc.sh <ckpt> [variant] [dataset] [limit_val_batches] [val_closed]
#   ckpt:              path to .ckpt file OR run directory (appends /checkpoints/last.ckpt)
#   variant:           smart (default), hmart, hmart_aux, hmart_c8, hmart_c32,
#                      hmart_intent, hmart_intent_aux, smart_intent_aux,
#                      hmart_c4_intent_aux, hmart_c8_intent_aux
#                      (MUST match the variant the checkpoint was trained with)
#   dataset:           adv (default), unbias
#   limit_val_batches: 0.1 (default) — same fraction used in sc_pre_bc.yaml so
#                      val_open metrics line up with BC pre-training logs.
#                      Use 50 to match sc_clsft.yaml's cap, or 1.0 for full val.
#   val_closed:        no (default), yes — closed-loop rollouts are ~128× more
#                      expensive than open-loop, so off by default. Turn on if
#                      you also want val_closed/ADE_{own,opp}_rollout numbers.
#
# Runs `action=validate` on the given checkpoint with a matching val protocol,
# so you can compare the number against the BC training run (0.1) or a CLSFT
# training run (50).
#
# Example (BC hmart_intent_aux ckpt, open-loop only, 0.1 val — same as BC logs):
#   sbatch scripts_rcac/run.sbatch eval_val_sc.sh \
#     logs/hmart_intent_aux-rcac/runs/2026-04-19_01-46-25/checkpoints/last.ckpt \
#     hmart_intent_aux

set -e

if [ -z "$1" ]; then
  echo "Error: missing checkpoint argument"
  echo "Usage: sbatch scripts_rcac/run.sbatch eval_val_sc.sh <ckpt> [variant] [dataset] [limit_val_batches] [val_closed]"
  exit 1
fi

CKPT_INPUT=$1
VARIANT=${2:-smart}
DATASET=${3:-adv}
LIMIT_VAL=${4:-0.1}
VAL_CLOSED=${5:-no}

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=${WANDB_MODE:-online}

MY_EXPERIMENT="sc_pre_bc"
MY_TASK_NAME="sc_val_${VARIANT}-rcac"

# --- Dataset ---
if [ "$DATASET" = "unbias" ]; then
  DATA_OVERRIDES="data=starcraft_unbias"
else
  DATA_OVERRIDES=""
fi

# --- Variant-specific model flags (must match the ckpt's training run) ---
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

# --- val_closed toggle ---
if [ "$VAL_CLOSED" = "yes" ]; then
  VAL_CLOSED_OVERRIDE="model.model_config.val_closed_loop=true"
else
  VAL_CLOSED_OVERRIDE="model.model_config.val_closed_loop=false"
fi

echo "=== scripts_rcac/eval_val_sc.sh | variant=$VARIANT | dataset=$DATASET | limit_val_batches=$LIMIT_VAL | val_closed=$VAL_CLOSED ==="
echo "    ckpt: $MY_CKPT_PATH"

# --- Launcher: torchrun+DDP on multi-GPU SLURM allocations, python on 1 GPU ---
NGPUS=${SLURM_GPUS_ON_NODE:-1}

if [ "$NGPUS" -gt 1 ]; then
  echo "    launcher: torchrun on $NGPUS GPUs (DDP, SLURM rendezvous)"
  torchrun \
    --rdzv_id ${SLURM_JOB_ID:-0} \
    --rdzv_backend c10d \
    --rdzv_endpoint ${MASTER_ADDR:-localhost}:${MASTER_PORT:-29500} \
    --nnodes ${SLURM_NNODES:-1} \
    --nproc_per_node $NGPUS \
    -m src.run \
    action=validate \
    experiment=$MY_EXPERIMENT \
    trainer=ddp \
    trainer.accelerator=gpu \
    trainer.devices=$NGPUS \
    trainer.limit_val_batches=$LIMIT_VAL \
    ckpt_path=$MY_CKPT_PATH \
    model.model_config.val_open_loop=true \
    $VAL_CLOSED_OVERRIDE \
    model.model_config.n_vis_batch=0 \
    task_name=$MY_TASK_NAME \
    $DATA_OVERRIDES \
    $MODEL_OVERRIDES
else
  echo "    launcher: python on 1 GPU"
  python -m src.run \
    action=validate \
    experiment=$MY_EXPERIMENT \
    trainer=default \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    trainer.strategy=auto \
    trainer.limit_val_batches=$LIMIT_VAL \
    ckpt_path=$MY_CKPT_PATH \
    model.model_config.val_open_loop=true \
    $VAL_CLOSED_OVERRIDE \
    model.model_config.n_vis_batch=0 \
    task_name=$MY_TASK_NAME \
    $DATA_OVERRIDES \
    $MODEL_OVERRIDES
fi

echo "scripts_rcac/eval_val_sc.sh done!"
