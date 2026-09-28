#!/bin/bash
#SBATCH --job-name=grouparc-lse
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -u

# sbatch should be invoked from the updated CompressARC checkout.
cd "${SLURM_SUBMIT_DIR}"

# Batch shells do not normally activate the interactive Conda environment.
if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
    echo "Could not find conda.sh under ${HOME}/miniconda3 or ${HOME}/anaconda3" >&2
    exit 1
fi

conda activate SymARC

python analyze_example_wandb.py --help | grep -q -- "--cummax-variant" || {
    echo "This is the old checkout: --cummax-variant is unavailable." >&2
    exit 1
}

tasks=(50cb2852 54d82841 623ea044 c1d99e64 ec883f72)
variants=(projected_lse pure_lse)
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
            --direction-share-variant original \
            --cummax-variant "${variant}" \
            --early-stop \
            --early-stop-window 50 \
            --early-stop-epsilon 1e-4 \
            --early-stop-warmup 300 \
            --wandb \
            --wandb-entity arc_agi \
            --wandb-project grouparc \
            --wandb-tags lse-ablation m3 \
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

echo "All requested LSE conditions completed."
