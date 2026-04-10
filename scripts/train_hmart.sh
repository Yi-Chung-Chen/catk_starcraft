#!/bin/bash
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Ablation: hmart (global concepts, no aux)

MY_EXPERIMENT="sc_pre_bc"
MY_TASK_NAME="hmart"

torchrun --nproc_per_node=gpu \
  -m src.run \
  experiment=$MY_EXPERIMENT \
  task_name=$MY_TASK_NAME \
  data.train_batch_size=8 \
  data.val_batch_size=8 \
  data.test_batch_size=8 \
  model.model_config.use_aux_loss=false \
  model.model_config.decoder.num_concepts=16

echo "train_hmart.sh done!"
