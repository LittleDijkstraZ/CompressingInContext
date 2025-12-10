#!/bin/bash
#SBATCH --job-name=eval_fold
#SBATCH --partition=h100,a100
#SBATCH --gpus=1
#SBATCH --exclude=c007,h02
#SBATCH --mem=80G
#SBATCH --time=24:00:00

# Arguments passed via environment:
# FOLD_CACHES: space-separated list of cache directories
# RESULTS_DIR: directory to save results
# PROJECT_DIR: project root directory

cd "$PROJECT_DIR"

export RKV_DEBUG_COMPRESSION=1

# Convert space-separated string back to array
IFS=' ' read -ra CACHES <<< "$FOLD_CACHES"

echo "Processing ${#CACHES[@]} caches in this fold"

for cache_dir in "${CACHES[@]}"; do
    cache_name=$(basename "$cache_dir")
    output_file="$RESULTS_DIR/results_${cache_name}.json"
    
    echo "=========================================="
    echo "Evaluating: $cache_name"
    echo "Output: $output_file"
    echo "=========================================="
    
    python -m src.pipeline.eval_cache \
        --model_name "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B" \
        --kv_cache_dir "$cache_dir" \
        --max_new_tokens 8192 \
        --repeat_time 16 \
        --output_file "$output_file"
    
    if [ $? -eq 0 ]; then
        echo "✓ Completed: $cache_name"
    else
        echo "✗ Failed: $cache_name"
    fi
done

echo "Fold evaluation complete!"

