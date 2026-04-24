#!/bin/bash
# Usage: bash scripts_rcac/train_sc_clsft.sh <bc_ckpt_path> [variant] [dataset] [resume_dir]
#   bc_ckpt_path: REQUIRED, e.g. logs/hmart_intent_aux-rcac/runs/2026-04-19_01-46-25/checkpoints/last.ckpt
#   variant: smart (default), hmart, hmart_aux, hmart_c8, hmart_c32,
#            hmart_intent, hmart_intent_aux, smart_intent_aux, hmart_c4_intent_aux,
#            hmart_c8_intent_aux  (MUST match the variant used for BC pre-training)
#   dataset: adv (default), unbias
#   resume_dir: optional, e.g. logs/hmart_intent_aux_clsft-rcac/runs/2026-04-25_12-00-00
#               When set, resumes CLSFT run from that dir's last.ckpt:
#                 - reuses hydra.run.dir + wandb id
#                 - flips action=fit (Lightning resumes optimizer/epoch state)
#
# RCAC-specific: batch_size=32 (CLSFT uses less activation memory than BC
# because rollout steps are sequential with detached positions), 1-GPU
# (single-GPU fit on A100 80GB) or DDP when SLURM_GPUS_ON_NODE>1.
# val_batch_size stays at 4 (val stitches n_rollout_closed_val=32 replicas).

export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BC_CKPT=${1:?"BC checkpoint path required as first arg (see header comment)"}
VARIANT=${2:-smart}
DATASET=${3:-adv}
RESUME_DIR=${4:-}

if [ ! -f "$BC_CKPT" ]; then
  echo "BC checkpoint not found: $BC_CKPT"; exit 1
fi

MY_EXPERIMENT="sc_clsft"
MY_TASK_NAME="${VARIANT}_clsft-rcac"
BATCH_SIZE=32

# --- Dataset ---
if [ "$DATASET" = "unbias" ]; then
  DATA_OVERRIDES="data=starcraft_unbias"
else
  DATA_OVERRIDES=""
fi

# --- Variant-specific model flags (MUST match the BC run) ---
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
    BATCH_SIZE=16
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

# --- Ckpt / resume handling ---
# Fresh CLSFT: action=finetune (from sc_clsft.yaml) + ckpt_path=$BC_CKPT
#   → strict=False load, epoch 0, fresh optimizer.
# Resume CLSFT: points at the CLSFT run's own last.ckpt, reuses hydra.run.dir
#   and wandb id, flips action=fit so Lightning resumes optimizer + epoch.
ACTION_OVERRIDE=""
if [ -n "$RESUME_DIR" ]; then
  if [ ! -d "$RESUME_DIR" ]; then
    echo "Resume dir not found: $RESUME_DIR"; exit 1
  fi
  RESUME_CKPT="$RESUME_DIR/checkpoints/last.ckpt"
  if [ ! -f "$RESUME_CKPT" ]; then
    echo "Resume checkpoint not found: $RESUME_CKPT"; exit 1
  fi
  RESUME_WANDB_FOLDER=$(ls -1d "$RESUME_DIR"/wandb/run-* 2>/dev/null | tail -n1)
  if [ -z "$RESUME_WANDB_FOLDER" ]; then
    echo "No wandb run folder under $RESUME_DIR/wandb/"; exit 1
  fi
  RESUME_WANDB_ID=$(basename "$RESUME_WANDB_FOLDER" | sed -E 's/^run-[0-9]+_[0-9]+-//')
  if [ -z "$RESUME_WANDB_ID" ]; then
    echo "Failed to parse wandb run id from $RESUME_WANDB_FOLDER"; exit 1
  fi
  RESUME_ABS_DIR=$(cd "$RESUME_DIR" && pwd)
  echo "  resume: dir=$RESUME_ABS_DIR"
  echo "  resume: ckpt=$RESUME_CKPT"
  echo "  resume: wandb_id=$RESUME_WANDB_ID"
  CKPT_OVERRIDES="hydra.run.dir=$RESUME_ABS_DIR ckpt_path=$RESUME_CKPT logger.wandb.id=$RESUME_WANDB_ID"
  ACTION_OVERRIDE="action=fit"
else
  CKPT_OVERRIDES="ckpt_path=$BC_CKPT"
fi

echo "=== train_sc_clsft.sh | variant=$VARIANT | dataset=$DATASET | batch=$BATCH_SIZE | bc_ckpt=$BC_CKPT${RESUME_DIR:+ | resume=$RESUME_DIR} ==="

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
    $DATA_OVERRIDES \
    $MODEL_OVERRIDES \
    $CKPT_OVERRIDES \
    $ACTION_OVERRIDE
else
  python -m src.run \
    experiment=$MY_EXPERIMENT \
    task_name=$MY_TASK_NAME \
    data.train_batch_size=$BATCH_SIZE \
    $DATA_OVERRIDES \
    $MODEL_OVERRIDES \
    $CKPT_OVERRIDES \
    $ACTION_OVERRIDE
fi

echo "train_sc_clsft.sh done!"
