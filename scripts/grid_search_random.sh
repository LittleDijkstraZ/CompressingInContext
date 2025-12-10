#!/bin/bash
#SBATCH --job-name=grid_search_random
#SBATCH --partition=h100,a100
#SBATCH --gpus=1
#SBATCH --exclude=c007,h02
#SBATCH --mem=80G
#SBATCH --output=work_dirs/slurm/grid_search_random/grid_search_random_%j.out
#SBATCH --error=work_dirs/slurm/grid_search_random/grid_search_random_%j.err
#SBATCH --time=48:00:00

export RKV_DEBUG_COMPRESSION=1

# ============================================================================
# Configuration
# ============================================================================

# Number of random configurations to run
NUM_CONFIGS=1024

# Random seed for reproducibility (comment out for random each time)
RANDOM_SEED=42

# Hyperparameter ranges (all possible values)
MODEL_NAMES="PlanePaper/LEAD-7B deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
BUDGETS="256 512 1024 2048 4096 8192 16384"
WINDOW_SIZES="32 64 128 256 384 512 786"
ROTATIONS="none 16 256 1024 2048 4096 8192"
DATA_LIMITS="1 2 3 5 8 13"
MODES="takeaways notepad none"
NUM_EPOCHS="1 2 3 5"


# Directories
CACHE_DIR=./kv_caches_random
RESULTS_DIR=./results_grid_search_random

# Config file path
CONFIG_FILE="./random_configs.json"

# ============================================================================
# Setup
# ============================================================================

# Create directories if they don't exist
mkdir -p "$CACHE_DIR"
mkdir -p "$RESULTS_DIR"
mkdir -p "$CONFIGS_DIR"

# ============================================================================
# Generate random configurations (or use existing)
# ============================================================================

if [ -f "$CONFIG_FILE" ]; then
    echo "=========================================="
    echo "Using existing configuration file: ${CONFIG_FILE}"
    echo "=========================================="
    echo "To generate new configurations, delete the file and re-run."
else
    echo "=========================================="
    echo "Generating random configurations"
    echo "=========================================="

    # Build the command for generating configs
    GEN_CMD="python scripts/generate_random_configs.py \
        --num-configs ${NUM_CONFIGS} \
        --output ${CONFIG_FILE} \
        --model-names ${MODEL_NAMES} \
        --budgets ${BUDGETS} \
        --window-sizes ${WINDOW_SIZES} \
        --rotations ${ROTATIONS} \
        --data-limits ${DATA_LIMITS} \
        --modes ${MODES} \
        --num-epochs ${NUM_EPOCHS}"

    # Add seed if defined
    if [ -n "$RANDOM_SEED" ]; then
        GEN_CMD="${GEN_CMD} --seed ${RANDOM_SEED}"
    fi

    echo "Running: ${GEN_CMD}"
    eval ${GEN_CMD}

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to generate configurations"
        exit 1
    fi
fi

# ============================================================================
# Run grid search with the generated configurations
# ============================================================================
echo ""
echo "=========================================="
echo "Running random grid search"
echo "=========================================="

# Read configurations from JSON and run each one
python -c "
import json
import subprocess
import sys

config_file = '${CONFIG_FILE}'
with open(config_file, 'r') as f:
    data = json.load(f)

configs = data['configs']
total = len(configs)
print(f'Running {total} configurations...')
print()

for i, config in enumerate(configs):
    print(f'==========================================')
    print(f'Configuration {i+1}/{total}')
    print(f'  model_name={config[\"model_name\"]}')
    print(f'  budget={config[\"budget\"]}')
    print(f'  window_size={config[\"window_size\"]}')
    print(f'  rotation={config[\"rotation\"]}')
    print(f'  data_limit={config[\"data_limit\"]}')
    print(f'  mode={config[\"mode\"]}')
    print(f'  num_epochs={config[\"num_epochs\"]}')
    print(f'==========================================')

    cmd = [
        'python', '-m', 'src.pipeline.run_grid_search_none',
        '--data-path', './src/clustering/similar_questions_result_aime25_3.json',
        '--cache-dir', '${CACHE_DIR}',
        '--results-dir', '${RESULTS_DIR}',
        '--model-names', config['model_name'],
        '--budget-range', str(config['budget']),
        '--window-sizes', str(config['window_size']),
        '--rotation-configs', str(config['rotation']),
        '--data-limits', str(config['data_limit']),
        '--modes', config['mode'],
        '--num-epochs-list', str(config['num_epochs']),
    ]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f'WARNING: Configuration {i+1} failed')
    print()

print('==========================================')
print('Random grid search complete!')
print('==========================================')
"

echo "=========================================="
echo "Random grid search complete!"
echo "=========================================="

