# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CAT-K (Closest Among Top-K) is a closed-loop supervised fine-tuning method for tokenized traffic simulation models, built on top of SMART. Accepted as Oral at CVPR 2025; ranked #1 on WOSAC leaderboard (2024). The core idea: during fine-tuning, at each timestep, take the top-K most likely action tokens and select the one that keeps the agent closest to ground truth.

## Commands

### Environment Setup
```bash
conda create -y -n catk python=3.11.9
conda activate catk
conda install -y -c conda-forge ffmpeg=4.3.2
pip install -r install/requirements.txt
pip install torch_geometric
pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install --no-deps waymo-open-dataset-tf-2-12-0==1.6.4
```

### Running Experiments

All runs go through `src/run.py` via Hydra. The `action` field controls mode (`fit`, `finetune`, `validate`, `test`).

**BC Pre-training (single GPU):**
```bash
torchrun -m src.run experiment=pre_bc task_name=my_task
```

**CAT-K Fine-tuning** (requires `ckpt_path` in `clsft.yaml` pointing to BC checkpoint):
```bash
torchrun -m src.run experiment=clsft task_name=my_task
```

**Local validation (single GPU):**
```bash
python -m src.run experiment=local_val trainer=default trainer.accelerator=gpu trainer.devices=1 trainer.strategy=auto model.model_config.validation_rollout_sampling.num_k=48 task_name=my_task
```

**Multi-GPU DDP:**
```bash
torchrun --rdzv_id $SLURM_JOB_ID --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT --nnodes $NUM_NODES --nproc_per_node gpu -m src.run experiment=pre_bc trainer=ddp task_name=my_task
```

**Dataset preprocessing:**
```bash
bash scripts/cache_womd.sh  # run for training, validation, and testing splits
```

**WOSAC submission packaging:**
```bash
bash scripts/wosac_sub.sh
```

### Overriding Config Values
Hydra allows CLI overrides. Examples:
```bash
python -m src.run experiment=pre_bc model=smart_nano_1M task_name=nano_run
python -m src.run experiment=clsft ckpt_path=/path/to/checkpoint.ckpt task_name=clsft_run
python -m src.run action=validate ckpt_path=/path/to/checkpoint.ckpt experiment=local_val
```

## Architecture

### Configuration System (Hydra)
Entry point: `configs/run.yaml`. Experiment configs under `configs/experiment/` override defaults. Key configs:
- `experiment/pre_bc.yaml` — behavior cloning pre-training (lr=5e-4, open-loop)
- `experiment/clsft.yaml` — CAT-K fine-tuning (lr=5e-5, closed-loop, `action=finetune`)
- `experiment/local_val.yaml` — local validation
- `experiment/ego_gmm_*.yaml` — GMM-based ego policy variants
- `model/smart.yaml` — 7M param model (hidden_dim=128, 8 heads)
- `model/smart_mini_3M.yaml`, `model/smart_nano_1M.yaml` — smaller variants
- `paths/default.yaml` — data cache path (`cache_root: /scratch/cache/SMART`)

### Model Architecture
Two model families:
1. **SMART** (`src/smart/model/smart.py`): The main traffic simulation model
2. **EgoGMMSMART** (`src/smart/model/ego_gmm_smart.py`): GMM-based ego policy variant

The decoder stack (`src/smart/modules/smart_decoder.py`) contains:
- **SMARTMapDecoder** (`modules/map_decoder.py`): Polyline-to-polyline attention (radius 10m)
- **SMARTAgentDecoder** (`modules/agent_decoder.py`):
  - Polyline-to-agent attention (radius 30m)
  - Agent-to-agent attention (radius 60m)
  - Temporal attention over 11 historical steps

### Tokenization (`src/smart/tokens/`)
- `token_processor.py`: Central class for converting trajectories ↔ tokens
  - Map tokens from `map_traj_token5.pkl`
  - Agent tokens from `agent_vocab_555_s2.pkl` (custom vocabulary, important for performance)
  - `agent_token_sampling.criterium` controls open-loop vs closed-loop token selection
  - `training_rollout_sampling.criterium=topk_prob_sampled_with_dist` with `temp=1e-5` implements CAT-K

### Data Pipeline
- `src/smart/datamodules/scalable_datamodule.py`: PyTorch Lightning DataModule for Waymo
- `src/smart/datamodules/target_builder.py`: Builds training targets from raw data
- `src/smart/datasets/scalable_dataset.py`: Dataset reading preprocessed pickle cache
- Raw data preprocessing: `src/data_preprocess.py`

### Metrics (`src/smart/metrics/`)
- `cross_entropy.py`: Token prediction loss
- `min_ade.py`: Minimum Average Displacement Error
- `next_token_cls.py`: Token classification accuracy
- `wosac_metrics.py`: Official WOSAC leaderboard metrics (RMM)
- `wosac_submission.py`: Packages submission tar.gz

### Utils (`src/smart/utils/`)
- `rollout.py`: Closed-loop rollout logic including CAT-K selection
- `finetune.py`: Fine-tuning utilities (freezing layers, etc.)
- `geometry.py`: Trajectory geometry utilities
- `preprocess.py`: Data preprocessing helpers

## StarCraft Branch (`src/starcraft/`)

### Overview
Adapts the SMART tokenized motion model for StarCraft II unit trajectory prediction. Uses the same token-based architecture but with StarCraft-specific data loading, map encoding, and auxiliary prediction heads.

### Model (`src/starcraft/model/sc_smart.py`)
- **SCSMART** LightningModule, config at `configs/model/sc_smart.yaml`
- `use_aux_loss: true/false` — controls whether auxiliary action/target prediction heads exist (not just loss weight — heads are removed when false). Ablate via `model.model_config.use_aux_loss=false`

### Data Pipeline
- **Dataset** (`src/starcraft/datasets/sc_dataset.py`): Reads per-replay HDF5 files from `StarCraftMotion_split_v2_medium/`
  - Loads `target_id` (uint32 row index, sentinel `0xFFFFFFFF`) and resolves to XY via coordinate lookup, merged into `target_pos` for ~96% target coverage
  - `target_id` and `target_pos` are mutually exclusive in the data
- **Token processor** (`src/starcraft/tokens/sc_token_processor.py`): Tokenizes agent trajectories and loads static map data
  - Map data cached per map_name (7 maps), loaded from `datasets/map_data/static/{Map_Name}.h5`
- **Target builder** (`src/starcraft/datamodules/sc_target_builder.py`): `train_mask` selects only **movable player units** (not static buildings, not neutrals) for the loss. All agents remain in the attention graph as context. No max_num cap — movable player unit counts are naturally manageable (~30-40 per scenario).

### Map Encoder (`src/starcraft/modules/sc_map_encoder.py`)
- CNN processes `[B, 3, 200, 200]` grid (pathing + height + creep) → `[B*625, hidden_dim]` patch tokens
- 25×25 patches (8×8 cells each), connected to agents via radius-based pl2a attention
- All 3 CNN input channels normalized to [-1, 1]:
  - **Pathing**: raw inverted first (1=walkable, 0=blocked), padded with 0 (blocked), then *2-1
  - **Height**: raw [0,255] → /255 to [0,1], padded with 0, then *2-1
  - **Creep**: raw {0,1} already padded with 0 (no creep), then *2-1

### Coordinate System (important)
- **Game coordinates**: X = column, Y increases upward from bottom
- **Pathing grid**: Row 0 = top of map = **high Y** in game coordinates
- **Pathing grid values**: SC2 raw API convention — **0 = walkable, 1 = blocked** (confirmed by placement grid overlap and neutral unit positions on pathing=1). Edges of map are all pathing=1.
- **Patch positions**: Y is flipped from row index: `Y = H - (row_idx * 8 + 4)`
- **Visualization**: `imshow` with `origin="upper"` and `extent=[0, W, 0, H]`

### Auxiliary Prediction Heads (when `use_aux_loss=true`)
Four heads in `sc_agent_decoder.py`: `has_action`, `has_target_pos`, `action_class` (11 classes), `target_pos` (2D XY)
- Loss in `src/starcraft/metrics/sc_action_target_loss.py`
- `decoder.use_action_target_input` feeds predictions back as input (requires `use_aux_loss=true`)

### Visualization (`src/starcraft/utils/vis_starcraft.py`)
- Animated GIFs showing GT + predicted trajectories
- Optional pathing grid background overlay (pre-padding map)
- Optional target position arrows (GT green "x", pred orange "x")
- Controlled by `n_vis_batch`, `n_vis_scenario`, `n_vis_rollout` in model config

### StarCraft Experiment Configs
- `experiment/sc_pre_bc.yaml` — BC pre-training (lr=5e-4, 64 epochs)
- `model/sc_smart.yaml` — default model (hidden_dim=64, 4 heads, 4 agent layers, 3 map layers)
- `data/starcraft.yaml` — StarCraft datamodule config

### Running StarCraft Experiments
```bash
# BC pre-training
torchrun -m src.run experiment=sc_pre_bc task_name=my_sc_task

# With aux loss ablation
torchrun -m src.run experiment=sc_pre_bc model.model_config.use_aux_loss=false task_name=no_aux

# Validation with visualization
python -m src.run experiment=sc_pre_bc action=validate ckpt_path=/path/to/ckpt.ckpt model.model_config.n_vis_batch=2
```

## Key Design Decisions

- **Two-phase training**: BC pre-training then CAT-K fine-tuning. The `action=finetune` mode loads weights with `load_state_dict(..., strict=False)` instead of resuming optimizer state.
- **Token vocabulary**: The `agent_vocab_555_s2.pkl` vocabulary is crucial — using the older `cluster_frame_5_2048_remove_duplicate.pkl` (from SMART) gives ~0.006 worse RMM.
- **CAT-K implementation**: In `clsft.yaml`, `training_rollout_sampling.criterium=topk_prob_sampled_with_dist` with `temp=1e-5` makes selection approximately deterministic (argmax over top-K by distance to GT).
- **Logging**: WandB is used for all logging. Set `WANDB_MODE=offline` for runs without internet.
- **Output**: Hydra writes logs and checkpoints to `logs/` directory, structured by task name and timestamp.
- **Scale**: SMART-tiny-7M needs 8x A100 (80GB); SMART-nano-1M can run on 1x A100 but is ~0.03 RMM worse.
