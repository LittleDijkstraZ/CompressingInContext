#!/bin/bash
#SBATCH --job-name=grid_search_tune
#SBATCH --partition=h100,a100
#SBATCH --gpus=1
#SBATCH --exclude=c007,h02
#SBATCH --mem=80G
#SBATCH --output=work_dirs/slurm/grid_search/grid_search_tune_%j.out
#SBATCH --error=work_dirs/slurm/grid_search/grid_search_tune_%j.err
#SBATCH --time=48:00:00

export RKV_DEBUG_COMPRESSION=1

model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# Default baseline parameters
DEFAULT_BUDGET=256
DEFAULT_WINDOW_SIZE=128
DEFAULT_ROTATION=256
DEFAULT_DATA_LIMIT=1
DEFAULT_MODE=takeaways
DEFAULT_NUM_EPOCHS=1

# Tunable values (excluding defaults - defaults are only run once at the start)
TUNE_BUDGETS="512 1024 2048 4096 8192"
TUNE_WINDOW_SIZES="32 64"
TUNE_ROTATIONS="none 16 1024 2048 4096 8192"
TUNE_DATA_LIMITS="2 3 5 8 13"
TUNE_MODES="notepad none"
TUNE_NUM_EPOCHS="3 5"

# Set directories (modify these as needed)
CACHE_DIR=./kv_caches_tune2
RESULTS_DIR=./results_grid_search_tune2

# Create directories if they don't exist
mkdir -p "$CACHE_DIR"
mkdir -p "$RESULTS_DIR"

# ============================================================================
# 1. Run the default configuration
# ============================================================================
echo "=========================================="
echo "Running default configuration"
echo "=========================================="
python -m src.pipeline.run_grid_search_none \
    --data-path ./src/clustering/similar_questions_result_aime25_3.json  \
    --cache-dir $CACHE_DIR \
    --results-dir $RESULTS_DIR \
    --model-names ${model_name} \
    --budget-range ${DEFAULT_BUDGET} \
    --window-sizes ${DEFAULT_WINDOW_SIZE} \
    --rotation-configs ${DEFAULT_ROTATION} \
    --data-limits ${DEFAULT_DATA_LIMIT} \
    --modes ${DEFAULT_MODE} \
    --num-epochs-list ${DEFAULT_NUM_EPOCHS}

# ============================================================================
# 2. Tune budget (fixing other parameters to defaults)
# ============================================================================
echo "=========================================="
echo "Tuning budget: ${TUNE_BUDGETS}"
echo "=========================================="
python -m src.pipeline.run_grid_search_none \
    --data-path ./src/clustering/similar_questions_result_aime25_3.json  \
    --cache-dir $CACHE_DIR \
    --results-dir $RESULTS_DIR \
    --model-names ${model_name} \
    --budget-range ${TUNE_BUDGETS} \
    --window-sizes ${DEFAULT_WINDOW_SIZE} \
    --rotation-configs ${DEFAULT_ROTATION} \
    --data-limits ${DEFAULT_DATA_LIMIT} \
    --modes ${DEFAULT_MODE} \
    --num-epochs-list ${DEFAULT_NUM_EPOCHS}

# ============================================================================
# 3. Tune window_size (fixing other parameters to defaults)
# ============================================================================
echo "=========================================="
echo "Tuning window_size: ${TUNE_WINDOW_SIZES}"
echo "=========================================="
python -m src.pipeline.run_grid_search_none \
    --data-path ./src/clustering/similar_questions_result_aime25_3.json  \
    --cache-dir $CACHE_DIR \
    --results-dir $RESULTS_DIR \
    --model-names ${model_name} \
    --budget-range ${DEFAULT_BUDGET} \
    --window-sizes ${TUNE_WINDOW_SIZES} \
    --rotation-configs ${DEFAULT_ROTATION} \
    --data-limits ${DEFAULT_DATA_LIMIT} \
    --modes ${DEFAULT_MODE} \
    --num-epochs-list ${DEFAULT_NUM_EPOCHS}

# ============================================================================
# 4. Tune rotation (fixing other parameters to defaults)
# ============================================================================
echo "=========================================="
echo "Tuning rotation: ${TUNE_ROTATIONS}"
echo "=========================================="
python -m src.pipeline.run_grid_search_none \
    --data-path ./src/clustering/similar_questions_result_aime25_3.json  \
    --cache-dir $CACHE_DIR \
    --results-dir $RESULTS_DIR \
    --model-names ${model_name} \
    --budget-range ${DEFAULT_BUDGET} \
    --window-sizes ${DEFAULT_WINDOW_SIZE} \
    --rotation-configs ${TUNE_ROTATIONS} \
    --data-limits ${DEFAULT_DATA_LIMIT} \
    --modes ${DEFAULT_MODE} \
    --num-epochs-list ${DEFAULT_NUM_EPOCHS}

# ============================================================================
# 5. Tune data_limit (fixing other parameters to defaults)
# ============================================================================
echo "=========================================="
echo "Tuning data_limit: ${TUNE_DATA_LIMITS}"
echo "=========================================="
python -m src.pipeline.run_grid_search_none \
    --data-path ./src/clustering/similar_questions_result_aime25_3.json  \
    --cache-dir $CACHE_DIR \
    --results-dir $RESULTS_DIR \
    --model-names ${model_name} \
    --budget-range ${DEFAULT_BUDGET} \
    --window-sizes ${DEFAULT_WINDOW_SIZE} \
    --rotation-configs ${DEFAULT_ROTATION} \
    --data-limits ${TUNE_DATA_LIMITS} \
    --modes ${DEFAULT_MODE} \
    --num-epochs-list ${DEFAULT_NUM_EPOCHS}

# ============================================================================
# 6. Tune mode (fixing other parameters to defaults)
# ============================================================================
echo "=========================================="
echo "Tuning mode: ${TUNE_MODES}"
echo "=========================================="
python -m src.pipeline.run_grid_search_none \
    --data-path ./src/clustering/similar_questions_result_aime25_3.json  \
    --cache-dir $CACHE_DIR \
    --results-dir $RESULTS_DIR \
    --model-names ${model_name} \
    --budget-range ${DEFAULT_BUDGET} \
    --window-sizes ${DEFAULT_WINDOW_SIZE} \
    --rotation-configs ${DEFAULT_ROTATION} \
    --data-limits ${DEFAULT_DATA_LIMIT} \
    --modes ${TUNE_MODES} \
    --num-epochs-list ${DEFAULT_NUM_EPOCHS}

# ============================================================================
# 7. Tune num_epochs (fixing other parameters to defaults)
# ============================================================================
echo "=========================================="
echo "Tuning num_epochs: ${TUNE_NUM_EPOCHS}"
echo "=========================================="
python -m src.pipeline.run_grid_search_none \
    --data-path ./src/clustering/similar_questions_result_aime25_3.json  \
    --cache-dir $CACHE_DIR \
    --results-dir $RESULTS_DIR \
    --model-names ${model_name} \
    --budget-range ${DEFAULT_BUDGET} \
    --window-sizes ${DEFAULT_WINDOW_SIZE} \
    --rotation-configs ${DEFAULT_ROTATION} \
    --data-limits ${DEFAULT_DATA_LIMIT} \
    --modes ${DEFAULT_MODE} \
    --num-epochs-list ${TUNE_NUM_EPOCHS}

echo "=========================================="
echo "Grid search tuning complete!"
echo "=========================================="

