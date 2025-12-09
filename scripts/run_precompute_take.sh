#!/bin/bash
# Loop through window_size range values and run precomputation for takeaways mode
# Fixed budget: 680

# Window size range
WINDOW=(64 128 256 384 512)

# Fixed budget
BUDGET=680

# Data path
DATA_PATH="./src/clustering/limo_clustering_results/k40/clusters/cluster_23.json"

# Mode
MODE="takeaways"

# GPU device
CUDA_DEVICE=9

# Base directory (project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "Running precomputation loop for window sizes"
echo "Mode: $MODE"
echo "Data: $DATA_PATH"
echo "Fixed Budget: $BUDGET"
echo "Window Sizes: ${WINDOW[@]}"
echo "=========================================="
echo ""

# Change to project root directory
cd "$PROJECT_ROOT"

# Loop through each window size
for WINDOW_SIZE in "${WINDOW[@]}"; do
    echo ""
    echo "=========================================="
    echo "Processing window_size: $WINDOW_SIZE (budget: $BUDGET)"
    echo "=========================================="
    
    LOG_FILE="./logs/precompute_${MODE}_budget${BUDGET}_window${WINDOW_SIZE}.log"
    
    # Run precomputation
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -m src.pipeline.precompute_cache_comp \
        --data_path "$DATA_PATH" \
        --mode "$MODE" \
        --budget "$BUDGET" \
        --window_size "$WINDOW_SIZE" \
        2>&1 | tee "$LOG_FILE"
    
    # Check exit status
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed window_size $WINDOW_SIZE (budget: $BUDGET)"
    else
        echo "✗ Failed for window_size $WINDOW_SIZE (budget: $BUDGET) (check $LOG_FILE for details)"
    fi
    
    echo ""
done

echo "=========================================="
echo "All window size runs completed!"
echo "=========================================="

