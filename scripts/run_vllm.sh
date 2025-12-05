#!/bin/bash
#SBATCH --job-name=seqr_finetune
#SBATCH --partition=h100,a100
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --output=work_dirs/slurm/finetune/finetune_seqr_%j.out
#SBATCH --error=work_dirs/slurm/finetune/finetune_seqr_%j.err
#SBATCH --time=48:00:00

python baseline_vllm.py \
    --model_name "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    --max_tokens 8192 \
    --n 32 \
    --output_file "results/my_baseline_2.json"