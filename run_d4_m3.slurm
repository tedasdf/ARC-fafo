#!/bin/bash
#SBATCH --job-name=grouparc-d4-no-es
#SBATCH --partition=gpu
#SBATCH --gres=gpu:T4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -u

# sbatch should be invoked from the updated CompressARC checkout.
cd "${SLURM_SUBMIT_DIR}"

module load miniforge3
eval "$(mamba shell hook --shell bash)"
mamba activate SymARC

python analyze_example_wandb.py --help | grep -q -- "--direction-share-variant" || {
    echo "This is the old checkout: --direction-share-variant is unavailable." >&2
    exit 1
}

tasks=(0dfd9992 22eb0ac0 239be575 50cb2852 54d82841 623ea044 c1d99e64 ec883f72)
variants=(projected_d4 pure_d4)
failures=()

for task in "${tasks[@]}"; do
    for variant in "${variants[@]}"; do
        echo "Starting ${task} / ${variant} at $(date --iso-8601=seconds)"

        if python analyze_example_wandb.py \
            --split training \
            --task "${task}" \
            --iterations 1500 \
            --multitensor-constraints strict \
            --shift-variant original \
            --direction-share-variant "${variant}" \
            --cummax-variant original \
            --wandb \
            --wandb-entity arc_agi \
            --wandb-project grouparc \
            --wandb-tags direction-share-ablation m3 no-early-stop \
            --skip-pca; then
            echo "Completed ${task} / ${variant} at $(date --iso-8601=seconds)"
        else
            status=$?
            failures+=("${task}/${variant}:${status}")
            echo "Failed ${task} / ${variant} with exit code ${status}; continuing." >&2
        fi
    done
done

if [ "${#failures[@]}" -gt 0 ]; then
    printf 'Failed conditions: %s\n' "${failures[*]}" >&2
    exit 1
fi

echo "All requested D4 conditions completed."
