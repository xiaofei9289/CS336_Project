#!/usr/bin/env bash
# Batch training launcher. Run from Assignment05 root.
# Examples:
#   export MODEL=/root/autodl-tmp/models/OLMo-2-0425-1B
#   export WANDB_PROJECT=cs336-a5-grpo
#   ./scripts/run_all_experiments.sh smoke          # 50-step smoke test
#   ./scripts/run_all_experiments.sh standard       # §4.3 four seeds
#   ./scripts/run_all_experiments.sh lr_sweep
#   ./scripts/run_all_experiments.sh prompt_ablation
#   ./scripts/run_all_experiments.sh variants        # §5.4
#   ./scripts/run_all_experiments.sh offpolicy       # §6.4
#   ./scripts/run_all_experiments.sh try_your_own    # §7
#   ./scripts/run_all_experiments.sh all             # everything (very long)
#
# Parallel jobs (separate GPUs + ports; do not reuse the same port):
#   VLLM_PORT=8000 POLICY_DEVICE=cuda:0 VLLM_GPU=1 ./scripts/run_all_experiments.sh standard &
#   VLLM_PORT=8001 POLICY_DEVICE=cuda:2 VLLM_GPU=3 SEEDS=2 ./scripts/run_all_experiments.sh standard &

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
MODEL="${MODEL:-allenai/OLMo-2-0425-1B}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT/results}"
SEEDS="${SEEDS:-0 1 2 3}"
STEPS="${STEPS:-200}"
LR="${LR:-1e-5}"
WANDB_PROJECT="${WANDB_PROJECT:-cs336-a5-grpo}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda:0}"
VLLM_GPU="${VLLM_GPU:-1}"
VLLM_PORT="${VLLM_PORT:-8000}"
EXTRA_ARGS=(${EXTRA_ARGS:-})

PROMPT_R1="$ROOT/cs336_alignment/prompts/r1_zero.prompt"
PROMPT_QO="$ROOT/cs336_alignment/prompts/question_only.prompt"
PROMPT_3S="$ROOT/cs336_alignment/prompts/r1_zero_three_shot_gsm8k.prompt"

train_standard() {
  local seed="$1"
  local out_name="$2"
  local prompt_path="$3"
  local lr="$4"
  local steps="$5"
  shift 5
  local out_dir="$RESULTS_ROOT/$out_name"
  echo "=== $out_name seed=$seed lr=$lr steps=$steps ==="
  "$PYTHON" scripts/train_grpo.py \
    --model "$MODEL" \
    --prompt "$prompt_path" \
    --seed "$seed" \
    --num-rollout-steps "$steps" \
    --learning-rate "$lr" \
    --policy-device "$POLICY_DEVICE" \
    --vllm-gpu "$VLLM_GPU" \
    --vllm-port "$VLLM_PORT" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "${out_name}_seed${seed}" \
    --output-dir "$out_dir" \
    "${EXTRA_ARGS[@]}" \
    "$@"
}

train_variant() {
  local recipe="$1"
  local seed="$2"
  local out_name="$3"
  local prompt_path="$4"
  local lr="$5"
  local steps="$6"
  shift 6
  local out_dir="$RESULTS_ROOT/$out_name"
  echo "=== $out_name seed=$seed lr=$lr steps=$steps ==="
  "$PYTHON" scripts/train_grpo_variants.py \
    --model "$MODEL" \
    --recipe "$recipe" \
    --prompt "$prompt_path" \
    --seed "$seed" \
    --num-rollout-steps "$steps" \
    --learning-rate "$lr" \
    --policy-device "$POLICY_DEVICE" \
    --vllm-gpu "$VLLM_GPU" \
    --vllm-port "$VLLM_PORT" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "${out_name}_seed${seed}" \
    --output-dir "$out_dir" \
    "${EXTRA_ARGS[@]}" \
    "$@"
}

run_smoke() {
  train_standard 0 smoke_seed0 "$PROMPT_R1" "$LR" 50
}

run_standard() {
  for seed in $SEEDS; do
    train_standard "$seed" "standard_seed_${seed}" "$PROMPT_R1" "$LR" "$STEPS"
  done
}

run_lr_sweep() {
  for lr in 5e-6 1e-5 3e-5; do
    for seed in $SEEDS; do
      train_standard "$seed" "lr_${lr}_seed_${seed}" "$PROMPT_R1" "$lr" "$STEPS"
    done
  done
}

run_prompt_ablation() {
  for seed in $SEEDS; do
    train_standard "$seed" "prompt_question_only_seed_${seed}" "$PROMPT_QO" "$LR" "$STEPS"
    train_standard "$seed" "prompt_r1_zero_three_shot_seed_${seed}" "$PROMPT_3S" "$LR" "$STEPS"
  done
}

run_variants() {
  local recipes=(grpo_constant dr_grpo rft maxrl)
  for recipe in "${recipes[@]}"; do
    for seed in $SEEDS; do
      train_variant "$recipe" "$seed" "${recipe}_seed_${seed}" "$PROMPT_R1" "$LR" "$STEPS"
    done
  done
}

run_offpolicy() {
  local recipes=(offpolicy_naive offpolicy_noclip offpolicy_clip offpolicy_gspo)
  for recipe in "${recipes[@]}"; do
    for seed in $SEEDS; do
      train_variant "$recipe" "$seed" "${recipe}_seed_${seed}" "$PROMPT_R1" "$LR" "$STEPS"
    done
  done
}

run_try_your_own() {
  for seed in $SEEDS; do
    train_variant try_your_own "$seed" "try_your_own_seed_${seed}" "$PROMPT_R1" "$LR" "$STEPS"
  done
}

MODE="${1:-smoke}"
case "$MODE" in
  smoke) run_smoke ;;
  standard) run_standard ;;
  lr_sweep) run_lr_sweep ;;
  prompt_ablation) run_prompt_ablation ;;
  variants) run_variants ;;
  offpolicy) run_offpolicy ;;
  try_your_own) run_try_your_own ;;
  all)
    run_standard
    run_lr_sweep
    run_prompt_ablation
    run_variants
    run_offpolicy
    run_try_your_own
    ;;
  *)
    echo "Unknown mode: $MODE"
    exit 1
    ;;
esac
