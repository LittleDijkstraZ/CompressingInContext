#!/bin/bash
#SBATCH --job-name=seqr_finetune
#SBATCH --partition=h100,a100
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --output=work_dirs/slurm/precompute/precompute_%j.out
#SBATCH --error=work_dirs/slurm/precompute/precompute_%j.err
#SBATCH --time=48:00:00

CUDA_VISIBLE_DEVICES=8 python -m src.pipeline.eval_cache \
    --kv_cache_dir ./hf_precomputed_kv_budget_680_comp_complex_takeaways \
    --repeat_time 16 

CUDA_VISIBLE_DEVICES=7 python -m src.pipeline.eval_cache \
    --kv_cache_dir ./hf_precomputed_kv_budget_680_comp_complex_takeaways \
    --repeat_time 16 \
    --follow_up_prompt "Three vertices of a cube are $P=(7,12,10)$ , $Q=(8,8,1)$ , and $R=(11,3,9)$ . What is the surface area of the cube?" \
    --output_file ./results/results_budget_ 680_comp_complex_takeaways_doc0.json

CUDA_VISIBLE_DEVICES=6 python -m src.pipeline.eval_cache \
    --kv_cache_dir ./hf_precomputed_kv_budget_680_comp_complex_notepad \
    --repeat_time 16 \
    --follow_up_prompt "Three vertices of a cube are $P=(7,12,10)$ , $Q=(8,8,1)$ , and $R=(11,3,9)$ . What is the surface area of the cube?" \
    --output_file ./results/results_budget_680_comp_complex_notepad_doc0.json