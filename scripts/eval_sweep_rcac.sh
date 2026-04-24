#!/bin/bash
# scripts/eval_sweep_rcac.sh
#
# Closed-loop eval sweep over the RCAC-trained checkpoints plus CV baselines.
# For every (variant, maps) pair we:
#   1. Run `src.run action=test` (or the CV generator) with
#      trainer.limit_test_batches capped so ~N_SCENARIOS replays are consumed,
#      writing rollouts into  <SWEEP_ROOT>/runs/<variant>_<maps>/rollouts/
#   2. Run scripts/eval_sc_rollouts.py on that rollouts dir, writing
#      <SWEEP_ROOT>/runs/<variant>_<maps>/metrics.csv
#   3. At the end, call scripts/aggregate_sweep_eval.py to concat every
#      per-run metrics.csv into <SWEEP_ROOT>/summary.csv + summary_table.txt
#
# Usage: bash scripts/eval_sweep_rcac.sh [n_scenarios] [batch_size]
#   n_scenarios: default 500. Converted to limit_test_batches = ceil(N/BS).
#   batch_size:  default 4. Passed as data.test_batch_size override so the
#                scenario count lands close to N_SCENARIOS (rounded up to the
#                nearest batch).
#
# Forces single-GPU so the scenario count is exactly limit_test_batches * BS
# (with DDP, limit_test_batches is per-rank, multiplying total scenarios).

set -euo pipefail

N_SCENARIOS=${1:-500}
BATCH_SIZE=${2:-4}
LIMIT_BATCHES=$(( (N_SCENARIOS + BATCH_SIZE - 1) / BATCH_SIZE ))

REPLAYS_DIR="datasets/StarCraftMotion_split_v2_adversarial"
SWEEP_TS="$(date +%Y-%m-%d_%H-%M-%S)"
SWEEP_ROOT="logs/rcac_sweep_eval/${SWEEP_TS}"
mkdir -p "$SWEEP_ROOT/runs"

export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export WANDB_MODE=${WANDB_MODE:-disabled}

if [ "${CONDA_DEFAULT_ENV:-}" != "catk" ]; then
  echo "Error: catk conda env not active. Run 'conda activate catk' first."
  exit 1
fi

# --- (variant, ckpt, model_overrides) rows ---------------------------------
# `variant` is the short name used in local_test_sc.sh, kept verbatim so the
# overrides stay drop-in compatible.
#
# For `hmart_intent` we want oracle intent input at test time — the model was
# trained with `use_action_target_input=true` and the eval-time oracle flag
# lives at `decoder.closed_loop_oracle_intent_input=true`. local_test_sc.sh's
# `hmart_intent` case already sets that, so we reuse it verbatim.
#
# Format: tag|ckpt|override_string
MODEL_ROWS=(
  "smart_intent_aux|logs_gautschi/smart_intent_aux-rcac/runs/2026-04-19_01-46-25/checkpoints/last.ckpt|model.model_config.use_aux_loss=true model.model_config.decoder.num_concepts=0 model.model_config.decoder.use_action_target_input=true"
  "hmart_intent_aux|logs_gautschi/hmart_intent_aux-rcac/runs/2026-04-19_01-46-25/checkpoints/last.ckpt|model.model_config.use_aux_loss=true model.model_config.decoder.num_concepts=16 model.model_config.decoder.use_action_target_input=true"
  "hmart_c4_intent_aux|logs_gautschi/hmart_c4_intent_aux-rcac/runs/2026-04-19_01-47-23/checkpoints/last.ckpt|model.model_config.use_aux_loss=true model.model_config.decoder.num_concepts=4 model.model_config.decoder.use_action_target_input=true"
  "hmart_c8_intent_aux|logs_gautschi/hmart_c8_intent_aux-rcac/runs/2026-04-19_01-47-12/checkpoints/last.ckpt|model.model_config.use_aux_loss=true model.model_config.decoder.num_concepts=8 model.model_config.decoder.use_action_target_input=true"
  "hmart|logs_gautschi/hmart-rcac/runs/2026-04-19_01-48-23/checkpoints/last.ckpt|model.model_config.use_aux_loss=false model.model_config.decoder.num_concepts=16"
  "hmart_intent|logs_gautschi/hmart_intent-rcac/runs/2026-04-19_01-48-23/checkpoints/last.ckpt|model.model_config.use_aux_loss=false model.model_config.decoder.num_concepts=16 model.model_config.decoder.use_action_target_input=true model.model_config.decoder.closed_loop_oracle_intent_input=true"
  "hmart_aux|logs_gautschi/hmart_aux-rcac/runs/2026-04-19_01-48-13/checkpoints/last.ckpt|model.model_config.use_aux_loss=true model.model_config.decoder.num_concepts=16"
)

# --- Map split overrides ----------------------------------------------------
declare -A MAP_OVERRIDE
MAP_OVERRIDE[id]='data.test_map_names=[Abyssal_Reef_LE,Acolyte_LE,Ascension_to_Aiur_LE,Interloper_LE,Mech_Depot_LE]'
MAP_OVERRIDE[ood]="data.test_map_names=['Catallena_LE_(Void)',Odyssey_LE]"

# --- Helpers ----------------------------------------------------------------
run_eval() {
  # Usage: run_eval <run_dir>
  local run_dir="$1"
  local rollouts_dir="$run_dir/rollouts"
  if [ ! -d "$rollouts_dir" ] || ! find "$rollouts_dir" -maxdepth 1 -name '*.h5' -print -quit | grep -q .; then
    echo "!! no rollouts produced under $rollouts_dir — skipping eval" | tee -a "$run_dir/eval.log"
    return 0
  fi
  python scripts/eval_sc_rollouts.py \
    --rollouts_dir "$rollouts_dir" \
    --replays_dir "$REPLAYS_DIR" \
    --out_csv "$run_dir/metrics.csv" \
    2>&1 | tee "$run_dir/eval.log"
}

run_model() {
  # Usage: run_model <variant> <ckpt> <maps> <model_overrides>
  local variant="$1"
  local ckpt="$2"
  local maps="$3"
  local overrides="$4"
  local tag="${variant}_${maps}"
  local run_dir="$SWEEP_ROOT/runs/$tag"
  mkdir -p "$run_dir"

  if [ ! -f "$ckpt" ]; then
    echo "!! ckpt missing, skipping: $ckpt" | tee -a "$run_dir/eval.log"
    return 0
  fi

  echo "=== [$tag] generate rollouts ==="
  # Single-GPU so limit_test_batches → exactly N_SCENARIOS scenarios total.
  python -m src.run \
    action=test \
    experiment=sc_pre_bc \
    trainer=default \
    trainer.devices=1 \
    trainer.accelerator=gpu \
    trainer.strategy=auto \
    trainer.limit_test_batches=$LIMIT_BATCHES \
    data.test_batch_size=$BATCH_SIZE \
    ckpt_path="$ckpt" \
    model.model_config.val_closed_loop=true \
    model.model_config.val_open_loop=false \
    model.model_config.n_rollout_closed_val=16 \
    model.model_config.n_vis_batch=0 \
    model.model_config.n_vis_scenario=0 \
    model.model_config.save_closed_rollouts=true \
    task_name="rcac_sweep_${tag}" \
    hydra.run.dir="$run_dir" \
    "${MAP_OVERRIDE[$maps]}" \
    $overrides \
    2>&1 | tee "$run_dir/test.log"

  echo "=== [$tag] eval metrics ==="
  run_eval "$run_dir"
}

run_cv() {
  # Usage: run_cv <tag> <maps> [+noise_vel_sigma=..]
  local tag="$1"
  local maps="$2"
  shift 2
  local extra_args=("$@")
  local run_dir="$SWEEP_ROOT/runs/$tag"
  mkdir -p "$run_dir"

  echo "=== [$tag] generate CV rollouts ==="
  python scripts/generate_constant_vel_rollouts.py \
    maps="$maps" \
    trainer.limit_test_batches=$LIMIT_BATCHES \
    data.test_batch_size=$BATCH_SIZE \
    model.model_config.n_rollout_closed_val=16 \
    hydra.run.dir="$run_dir" \
    task_name="rcac_sweep_${tag}" \
    "${extra_args[@]}" \
    2>&1 | tee "$run_dir/test.log"

  echo "=== [$tag] eval metrics ==="
  run_eval "$run_dir"
}

# --- Main sweep -------------------------------------------------------------
echo "Sweep root: $SWEEP_ROOT"
echo "limit_test_batches=$LIMIT_BATCHES  (≈ $((LIMIT_BATCHES * BATCH_SIZE)) scenarios per run)"

for maps in id ood; do
  for row in "${MODEL_ROWS[@]}"; do
    IFS='|' read -r variant ckpt overrides <<< "$row"
    run_model "$variant" "$ckpt" "$maps" "$overrides"
  done

  run_cv "cv_${maps}" "$maps"
  run_cv "cv_noise_${maps}" "$maps" "+noise_vel_sigma=0.06"
done

# --- Aggregate --------------------------------------------------------------
echo "=== aggregate ==="
python scripts/aggregate_sweep_eval.py --sweep_root "$SWEEP_ROOT"

echo
echo "Sweep complete. Results:"
echo "  per-run: $SWEEP_ROOT/runs/<variant>_<maps>/metrics.csv"
echo "  summary: $SWEEP_ROOT/summary.csv"
echo "  table:   $SWEEP_ROOT/summary_table.txt"
