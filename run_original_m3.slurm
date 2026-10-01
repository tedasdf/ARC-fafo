#!/bin/bash
#SBATCH --job-name=grouparc-original-no-es
#SBATCH --partition=gpu
#SBATCH --gres=gpu:T4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -eu

cd "${SLURM_SUBMIT_DIR}"

module load miniforge3
eval "$(mamba shell hook --shell bash)"
mamba activate SymARC

tasks=(0dfd9992 22eb0ac0 239be575)
failures=()

for task in "${tasks[@]}"; do
    echo "Starting ${task} / original at $(date --iso-8601=seconds)"

    if python analyze_example_wandb.py \
        --split training \
        --task "${task}" \
        --iterations 1500 \
        --multitensor-constraints strict \
        --shift-variant original \
        --direction-share-variant original \
        --cummax-variant original \
        --wandb \
        --wandb-entity arc_agi \
        --wandb-project grouparc \
        --wandb-tags original-baseline m3 no-early-stop \
        --skip-pca; then
        echo "Completed ${task} / original at $(date --iso-8601=seconds)"
    else
        status=$?
        failures+=("${task}/original:${status}")
        echo "Failed ${task} / original with exit code ${status}; continuing." >&2
    fi
done

if [ "${#failures[@]}" -gt 0 ]; then
    printf 'Failed conditions: %s\n' "${failures[*]}" >&2
    exit 1
fi

echo "All requested original conditions completed."
