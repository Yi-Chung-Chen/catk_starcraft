#!/bin/sh
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------------------------------------------------------------
# StarCraft Motion — BC Pre-training
#
# Uses experiment=sc_pre_bc which sets:
#   data=starcraft          (SCDataModule, HDF5 scenario files)
#   model=sc_smart          (SCSMART, 128-dim / 8-head / 6-layer)
#   action=fit
#   lr=5e-4, max_epochs=64
#
# Paths (edit in configs/paths/default.yaml or override here):
#   sc_dataset_root  — root of StarCraftMotion_split_v1_medium
#   sc_motion_dict   — path to motion_dictionary.pkl
# ---------------------------------------------------------------

MY_EXPERIMENT="sc_pre_bc"
MY_TASK_NAME=$MY_EXPERIMENT"-debug"

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
  model.model_config.decoder.num_concepts=0

# Multi-GPU DDP (uncomment for multi-GPU or multi-node):
# torchrun \
#   --rdzv_id $SLURM_JOB_ID \
#   --rdzv_backend c10d \
#   --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
#   --nnodes $NUM_NODES \
#   --nproc_per_node gpu \
#   -m src.run \
#   experiment=$MY_EXPERIMENT \
#   trainer=ddp \
#   task_name=$MY_TASK_NAME

echo "bash train_sc.sh done!"
