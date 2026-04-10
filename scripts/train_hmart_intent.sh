#!/bin/sh
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------------------------------------------------------------
# StarCraft Motion — hmart_intent
# Global concepts + intent input, no aux loss
# ---------------------------------------------------------------

MY_EXPERIMENT="sc_pre_bc"
MY_TASK_NAME="hmart_intent"

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate catk

# Single-GPU
torchrun --nproc_per_node=gpu \
  -m src.run \
  experiment=$MY_EXPERIMENT \
  task_name=$MY_TASK_NAME \
  data.train_batch_size=8 \
  data.val_batch_size=8 \
  data.test_batch_size=8 \
  trainer.max_epochs=10 \
  model.model_config.use_aux_loss=false \
  model.model_config.decoder.num_concepts=16 \
  model.model_config.decoder.use_action_target_input=true \
  model.model_config.decoder.closed_loop_oracle_intent_input=true

echo "train_hmart_intent.sh done!"
