#!/bin/bash
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Ablation: hmart_aux (global concepts + aux loss)

MY_EXPERIMENT="sc_pre_bc"
MY_TASK_NAME="hmart_aux"

torchrun --nproc_per_node=gpu \
  -m src.run \
  experiment=$MY_EXPERIMENT \
  task_name=$MY_TASK_NAME \
  data.train_batch_size=8 \
  data.val_batch_size=8 \
  data.test_batch_size=8 \
  trainer.max_epochs=10 \
  model.model_config.use_aux_loss=true \
  model.model_config.decoder.num_concepts=16

echo "train_hmart_aux.sh done!"
