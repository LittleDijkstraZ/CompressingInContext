#!/bin/bash
#SBATCH --job-name=seqr_finetune
#SBATCH --partition=h100,a100
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --output=work_dirs/slurm/precompute/precompute_%j.out
#SBATCH --error=work_dirs/slurm/precompute/precompute_%j.err
#SBATCH --time=48:00:00

python -m src.pipeline.eval_cache \
    --kv_cache_dir ./hf_precomputed_kv_budget_680_comp_complex \
    --repeat_time 16


# python -m src.pipeline.eval_cache     --kv_cache_dir ./hf_precomputed_kv_budget_780_comp_complex     --repeat_time 16 --model_name 'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B'