#!/bin/bash
# Loop through window_size values and run eval_cache for each KV cache
# Fixed budget: 680
#
# Usage:
#   ./run_eval_window_take.sh [--cuda-device DEVICE] [--windows W1 W2 W3 ...]
#   ./run_eval_window_take.sh --cuda-device 7 --windows 64 128 256 384 512
#   ./run_eval_window_take.sh --windows 64 128 256  # uses default CUDA_DEVICE=7

# Default values
CUDA_DEVICE=7
WINDOW=(64 128 256 384 512)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda-device)
            CUDA_DEVICE="$2"
            shift 2
            ;;
        --windows)
            shift
            WINDOW=()
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                WINDOW+=("$1")
                shift
            done
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --cuda-device DEVICE    GPU device ID (default: 7)"
            echo "  --windows W1 W2 ...     Window sizes to evaluate (default: 64 128 256 384 512)"
            echo "  --help, -h              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --cuda-device 7 --windows 64 128 256"
            echo "  $0 --windows 64 128 256 384 512"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Fixed budget
BUDGET=680

# Mode and complexity (should match precomputation settings)
MODE="takeaways"
COMPLEXITY="complex"

# Follow-up prompt for evaluation
FOLLOW_UP_PROMPT="Three vertices of a cube are \$P=(7,12,10)\$ , \$Q=(8,8,1)\$ , and \$R=(11,3,9)\$ . What is the surface area of the cube?"

# Repeat time
REPEAT_TIME=16

# Base directory (project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "Running eval_cache loop for window sizes"
echo "Fixed Budget: $BUDGET"
echo "Mode: $MODE"
echo "Complexity: $COMPLEXITY"
echo "Window Sizes: ${WINDOW[@]}"
echo "=========================================="
echo ""

# Change to project root directory
cd "$PROJECT_ROOT"

# Loop through each window size
for WINDOW_SIZE in "${WINDOW[@]}"; do
    echo ""
    echo "=========================================="
    echo "Evaluating window_size: $WINDOW_SIZE (budget: $BUDGET)"
    echo "=========================================="
    
    # Construct KV cache directory name
    KV_CACHE_DIR="./hf_precomputed_kv_budget_${BUDGET}__window_${WINDOW_SIZE}_comp_${COMPLEXITY}_${MODE}"
    
    # Check if KV cache directory exists
    if [ ! -d "$KV_CACHE_DIR" ]; then
        echo "⚠ Warning: KV cache directory not found: $KV_CACHE_DIR"
        echo "Skipping window_size $WINDOW_SIZE"
        continue
    fi
    
    # Output file name
    OUTPUT_FILE="./results/results_budget_${BUDGET}_window_${WINDOW_SIZE}_comp_${COMPLEXITY}_${MODE}_doc0.json"
    
    # Run eval_cache
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -m src.pipeline.eval_cache \
        --kv_cache_dir "$KV_CACHE_DIR" \
        --repeat_time $REPEAT_TIME \
        --follow_up_prompt "$FOLLOW_UP_PROMPT" \
        --output_file "$OUTPUT_FILE"     
        
    # Check exit status
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed eval for window_size $WINDOW_SIZE (budget: $BUDGET)"
    else
        echo "✗ Failed for window_size $WINDOW_SIZE (budget: $BUDGET) (check $LOG_FILE for details)"
    fi
    
    echo ""
done

echo "=========================================="
echo "All window size evaluations completed!"
echo "=========================================="

