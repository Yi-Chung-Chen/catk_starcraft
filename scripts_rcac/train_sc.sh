#!/bin/bash
# Usage: bash scripts_rcac/train_sc.sh [variant] [dataset] [resume_dir]
#   variant: smart (default), hmart, hmart_aux, hmart_c8, hmart_c32,
#            hmart_intent, hmart_intent_aux, smart_intent_aux
#   dataset: adv (default), unbias
#   resume_dir: optional, e.g. logs/smart_intent_aux-rcac/runs/2026-04-16_03-16-47
#               When set, resumes training from that run's checkpoints/last.ckpt,
#               writes new logs/checkpoints back into the same directory, and
#               continues the same wandb run (no new run created).

export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

VARIANT=${1:-smart}
DATASET=${2:-adv}
RESUME_DIR=${3:-}

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

# --- Resume (optional) ---
# When RESUME_DIR is given, we (1) override hydra.run.dir so new logs and
# checkpoints continue writing into the same directory, (2) point ckpt_path at
# the existing last.ckpt, and (3) reuse the existing wandb run id so the
# resumed steps stream into the same wandb run (resume: allow is already set
# in configs/logger/wandb.yaml).
RESUME_OVERRIDES=""
if [ -n "$RESUME_DIR" ]; then
  if [ ! -d "$RESUME_DIR" ]; then
    echo "Resume dir not found: $RESUME_DIR"; exit 1
  fi
  RESUME_CKPT="$RESUME_DIR/checkpoints/last.ckpt"
  if [ ! -f "$RESUME_CKPT" ]; then
    echo "Resume checkpoint not found: $RESUME_CKPT"; exit 1
  fi
  # Wandb persists each run under <save_dir>/wandb/run-<timestamp>-<id>/
  RESUME_WANDB_FOLDER=$(ls -1d "$RESUME_DIR"/wandb/run-* 2>/dev/null | tail -n1)
  if [ -z "$RESUME_WANDB_FOLDER" ]; then
    echo "No wandb run folder under $RESUME_DIR/wandb/"; exit 1
  fi
  RESUME_WANDB_ID=$(basename "$RESUME_WANDB_FOLDER" | sed -E 's/^run-[0-9]+_[0-9]+-//')
  if [ -z "$RESUME_WANDB_ID" ]; then
    echo "Failed to parse wandb run id from $RESUME_WANDB_FOLDER"; exit 1
  fi
  # Hydra needs an absolute path (it doesn't cd into run.dir before resolving).
  RESUME_ABS_DIR=$(cd "$RESUME_DIR" && pwd)
  echo "  resume: dir=$RESUME_ABS_DIR"
  echo "  resume: ckpt=$RESUME_CKPT"
  echo "  resume: wandb_id=$RESUME_WANDB_ID"
  RESUME_OVERRIDES="hydra.run.dir=$RESUME_ABS_DIR ckpt_path=$RESUME_CKPT logger.wandb.id=$RESUME_WANDB_ID"
fi

echo "=== train_sc.sh | variant=$VARIANT | dataset=$DATASET | batch=$BATCH_SIZE${RESUME_DIR:+ | resume=$RESUME_DIR} ==="

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
    trainer.max_epochs=16 \
    $DATA_OVERRIDES \
    $MODEL_OVERRIDES \
    $RESUME_OVERRIDES
else
  python -m src.run \
    experiment=$MY_EXPERIMENT \
    task_name=$MY_TASK_NAME \
    data.train_batch_size=$BATCH_SIZE \
    data.val_batch_size=$BATCH_SIZE \
    data.test_batch_size=$BATCH_SIZE \
    trainer.max_epochs=16 \
    $DATA_OVERRIDES \
    $MODEL_OVERRIDES \
    $RESUME_OVERRIDES
fi

echo "train_sc.sh done!"
