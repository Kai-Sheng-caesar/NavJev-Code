#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
if [[ -f .env ]]; then
    task_env_snapshot="$(export -p)"
    set -a
    source .env
    set +a
    eval "$task_env_snapshot"
    unset task_env_snapshot
fi
: "${TYPESAFE_API_KEY:?Set TYPESAFE_API_KEY in .env or the shell}"
export NAVJEV_CAPTION_MODEL="${NAVJEV_CAPTION_MODEL:-Salesforce/blip-image-captioning-base}"
export RAM_ROOT="${RAM_ROOT:-$REPO_ROOT/third_party/recognize-anything}"
HABITAT_CORE_ROOT="${HABITAT_CORE_ROOT:-$REPO_ROOT/third_party/habitat-core/habitat-lab}"
HABITAT_BASELINES_ROOT="${HABITAT_BASELINES_ROOT:-$REPO_ROOT/third_party/habitat-baselines}"
compat_root="$REPO_ROOT/logs/habitat_compat"
python_bin="${NAVJEV_PYTHON:-python}"
for package_path in "$HABITAT_CORE_ROOT/habitat" "$HABITAT_BASELINES_ROOT/habitat_baselines" "$RAM_ROOT/ram"; do
    if [[ ! -d "$package_path" ]]; then
        echo "Missing dependency: $package_path. Run python scripts/setup_dependencies.py or configure its root in .env." >&2
        exit 1
    fi
done
mkdir -p "$compat_root" logs/running_log logs/eval_results logs/tensorboard_dirs logs/checkpoints
ln -sfn "$HABITAT_CORE_ROOT/habitat" "$compat_root/habitat"
ln -sfn "$HABITAT_BASELINES_ROOT/habitat_baselines" "$compat_root/habitat_baselines"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$REPO_ROOT:$compat_root:$RAM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" scripts/check_setup.py --split "${NAVJEV_SPLIT:-rand100}"
exec "$python_bin" run.py \
    --exp_name "${NAVJEV_EXP_NAME:-evaluation_jev}" \
    --run-type eval --exp-config run_VLNBERT.yaml \
    SIMULATOR_GPU_IDS '[0]' TORCH_GPU_ID 0 TORCH_GPU_IDS '[0]' \
    EVAL.SPLIT "${NAVJEV_SPLIT:-rand100}" "$@"
