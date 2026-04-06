#!/bin/sh
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2

MY_EXPERIMENT="sc_pre_bc"
MY_TASK_NAME="sc_closed_val_eval"
# MY_CKPT_PATH="logs/sc_pre_bc-debug/runs/2026-03-26_00-54-15/checkpoints/last.ckpt"
# MY_CKPT_PATH="logs/sc_pre_bc-debug/runs/2026-03-30_22-56-07/checkpoints/last.ckpt"
MY_CKPT_PATH="logs/sc_pre_bc-debug/runs/2026-04-03_18-42-49/checkpoints/last.ckpt"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate catk

# Local StarCraft closed-loop validation on a single GPU.
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
  model.model_config.n_vis_batch=125 \
  model.model_config.n_vis_scenario=4 \
  trainer.limit_val_batches=125 \
  task_name=$MY_TASK_NAME

echo "bash local_val_sc.sh done!"
