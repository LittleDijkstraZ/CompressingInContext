#!/bin/bash
#SBATCH --job-name=grid_search
#SBATCH --partition=h100,a100
#SBATCH --gpus=1
#SBATCH --exclude=c007,h02
#SBATCH --mem=80G
#SBATCH --output=work_dirs/slurm/grid_search/grid_search_%j.out
#SBATCH --error=work_dirs/slurm/grid_search/grid_search_%j.err
#SBATCH --time=48:00:00

export RKV_DEBUG_COMPRESSION=1

# Set directories (modify these as needed)
CACHE_DIR="${CACHE_DIR:-./kv_caches_none}"
RESULTS_DIR="${RESULTS_DIR:-./results_grid_search_none}"

# Create directories if they don't exist
mkdir -p "$CACHE_DIR"
mkdir -p "$RESULTS_DIR"

cd CompressingInContext

python -m src.pipeline.run_grid_search_none \
    --data-path ./src/clustering/similar_questions_result.json \
    --cache-dir "$CACHE_DIR" \
    --results-dir "$RESULTS_DIR" \
    --rotation-configs none
