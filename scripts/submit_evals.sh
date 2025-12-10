#!/bin/bash
# Submit evaluation jobs for precomputed KV caches, divided into N folds
# Usage: ./submit_evals.sh [NUM_FOLDS] [KV_CACHE_DIR] [RESULTS_DIR]

set -e

# Configuration
NUM_FOLDS="${1:-4}"
KV_CACHE_DIR="${2:-./kv_caches}"
RESULTS_DIR="${3:-./results_evals}"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
JOB_SCRIPT="$SCRIPT_DIR/eval_fold.sh"

# Verify job script exists
if [ ! -f "$JOB_SCRIPT" ]; then
    echo "Error: Job script not found: $JOB_SCRIPT"
    exit 1
fi

# Create results directory
mkdir -p "$RESULTS_DIR"
mkdir -p "work_dirs/slurm/evals"

# Get all cache directories
mapfile -t CACHE_DIRS < <(find "$KV_CACHE_DIR" -maxdepth 1 -type d -name "hf_precomputed_kv_*" | sort)

NUM_CACHES=${#CACHE_DIRS[@]}
echo "Found $NUM_CACHES cache directories in $KV_CACHE_DIR"

if [ "$NUM_CACHES" -eq 0 ]; then
    echo "No cache directories found. Exiting."
    exit 1
fi

# Calculate fold sizes
FOLD_SIZE=$(( (NUM_CACHES + NUM_FOLDS - 1) / NUM_FOLDS ))
echo "Dividing into $NUM_FOLDS folds with ~$FOLD_SIZE caches each"

# Submit jobs for each fold
for ((fold=0; fold<NUM_FOLDS; fold++)); do
    start_idx=$((fold * FOLD_SIZE))
    end_idx=$((start_idx + FOLD_SIZE))
    
    # Ensure we don't go past the array
    if [ "$end_idx" -gt "$NUM_CACHES" ]; then
        end_idx=$NUM_CACHES
    fi
    
    # Skip empty folds
    if [ "$start_idx" -ge "$NUM_CACHES" ]; then
        echo "Fold $((fold+1)): No caches to process, skipping"
        continue
    fi
    
    # Get caches for this fold
    FOLD_CACHES=""
    for ((i=start_idx; i<end_idx; i++)); do
        if [ -n "$FOLD_CACHES" ]; then
            FOLD_CACHES="$FOLD_CACHES ${CACHE_DIRS[$i]}"
        else
            FOLD_CACHES="${CACHE_DIRS[$i]}"
        fi
    done
    
    num_in_fold=$((end_idx - start_idx))
    echo "Fold $((fold+1)): Submitting job for $num_in_fold caches (indices $start_idx to $((end_idx-1)))"
    
    # Submit job with environment variables
    sbatch \
        --job-name="eval_fold_$((fold+1))" \
        --output="work_dirs/slurm/evals/eval_fold_$((fold+1))_%j.out" \
        --error="work_dirs/slurm/evals/eval_fold_$((fold+1))_%j.err" \
        --export=ALL,FOLD_CACHES="$FOLD_CACHES",RESULTS_DIR="$RESULTS_DIR",PROJECT_DIR="$PROJECT_DIR" \
        "$JOB_SCRIPT"
done

echo ""
echo "Submitted $NUM_FOLDS evaluation jobs"
echo "Results will be saved to: $RESULTS_DIR"
