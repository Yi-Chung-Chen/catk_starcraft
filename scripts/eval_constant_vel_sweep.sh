#!/bin/bash
# scripts/eval_constant_vel_sweep.sh
#
# Closed-loop eval sweep for the StarCraft constant-velocity baselines.
# For every (baseline, maps) pair we:
#   1. Run scripts/generate_constant_vel_rollouts.py with
#      trainer.limit_test_batches capped so ~N_SCENARIOS replays are consumed,
#      writing rollouts into <SWEEP_ROOT>/runs/<baseline>_<maps>/rollouts/
#   2. Run scripts/eval_sc_rollouts.py on that rollouts dir, writing
#      <SWEEP_ROOT>/runs/<baseline>_<maps>/metrics.csv
#   3. At the end, call scripts/aggregate_sweep_eval.py to concat every
#      per-run metrics.csv into <SWEEP_ROOT>/summary.csv + summary_table.txt
#
# Baselines covered:
#   - cv: deterministic constant velocity
#   - cv_noise: constant velocity + Gaussian velocity noise
#
# Map splits covered:
#   - id
#   - ood
#
# Usage: bash scripts/eval_constant_vel_sweep.sh [n_scenarios] [batch_size]
#   n_scenarios: default all. Pass an integer to cap the sweep to ~N scenarios;
#                converted to limit_test_batches = ceil(N/BS).
#   batch_size:  default 4. Passed as data.test_batch_size override so the
#                scenario count lands close to N_SCENARIOS (rounded up to the
#                nearest batch).

set -euo pipefail

N_SCENARIOS=${1:-all}
BATCH_SIZE=${2:-4}
N_ROLLOUTS=32

LIMIT_BATCHES=""
if [ "$N_SCENARIOS" != "all" ]; then
  LIMIT_BATCHES=$(( (N_SCENARIOS + BATCH_SIZE - 1) / BATCH_SIZE ))
fi

REPLAYS_DIR="datasets/StarCraftMotion_split_v2_adversarial"
SWEEP_TS="$(date +%Y-%m-%d_%H-%M-%S)"
SWEEP_ROOT="logs/constant_vel_sweep/${SWEEP_TS}"
mkdir -p "$SWEEP_ROOT/runs"

export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export WANDB_MODE=${WANDB_MODE:-disabled}

if [ "${CONDA_DEFAULT_ENV:-}" != "catk" ]; then
  echo "Error: catk conda env not active. Run 'conda activate catk' first."
  exit 1
fi

run_eval() {
  # Usage: run_eval <run_dir>
  local run_dir="$1"
  local rollouts_dir="$run_dir/rollouts"
  if [ ! -d "$rollouts_dir" ] || ! find "$rollouts_dir" -maxdepth 1 -name '*.h5' -print -quit | grep -q .; then
    echo "!! no rollouts produced under $rollouts_dir -- skipping eval" | tee -a "$run_dir/eval.log"
    return 0
  fi
  python scripts/eval_sc_rollouts.py \
    --rollouts_dir "$rollouts_dir" \
    --replays_dir "$REPLAYS_DIR" \
    --out_csv "$run_dir/metrics.csv" \
    2>&1 | tee "$run_dir/eval.log"
}

run_cv() {
  # Usage: run_cv <tag> <maps> [extra generator args...]
  local tag="$1"
  local maps="$2"
  shift 2
  local extra_args=("$@")
  local run_dir="$SWEEP_ROOT/runs/$tag"
  local generator_args=(
    maps="$maps"
    "data.test_batch_size=$BATCH_SIZE"
    "model.model_config.n_rollout_closed_val=$N_ROLLOUTS"
    "hydra.run.dir=$run_dir"
    "task_name=constant_vel_sweep_${tag}"
  )
  mkdir -p "$run_dir"

  if [ -n "$LIMIT_BATCHES" ]; then
    generator_args+=("trainer.limit_test_batches=$LIMIT_BATCHES")
  fi

  echo "=== [$tag] generate CV rollouts ==="
  python scripts/generate_constant_vel_rollouts.py \
    "${generator_args[@]}" \
    "${extra_args[@]}" \
    2>&1 | tee "$run_dir/test.log"

  echo "=== [$tag] eval metrics ==="
  run_eval "$run_dir"
}

echo "Sweep root: $SWEEP_ROOT"
if [ -n "$LIMIT_BATCHES" ]; then
  echo "limit_test_batches=$LIMIT_BATCHES  (approx $((LIMIT_BATCHES * BATCH_SIZE)) scenarios per run)"
else
  echo "limit_test_batches=all  (full test split per run)"
fi
echo "n_rollouts=$N_ROLLOUTS"

for maps in id ood; do
  run_cv "cv_${maps}" "$maps"
  run_cv "cv_noise_${maps}" "$maps" "+noise_vel_sigma=0.06"
done

echo "=== aggregate ==="
python scripts/aggregate_sweep_eval.py --sweep_root "$SWEEP_ROOT"

echo
echo "Sweep complete. Results:"
echo "  per-run: $SWEEP_ROOT/runs/<baseline>_<maps>/metrics.csv"
echo "  summary: $SWEEP_ROOT/summary.csv"
echo "  table:   $SWEEP_ROOT/summary_table.txt"
