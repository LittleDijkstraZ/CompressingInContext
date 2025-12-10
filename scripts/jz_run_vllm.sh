#!/bin/bash
#SBATCH --job-name=seqr_finetune
#SBATCH --partition=h100,a100
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --output=work_dirs/slurm/finetune/finetune_seqr_%j.out
#SBATCH --error=work_dirs/slurm/finetune/finetune_seqr_%j.err
#SBATCH --time=48:00:00
# problem='For each positive integer $k,$ let $S_k$ denote the increasing arithmetic sequence of integers whose first term is 1 and whose common difference is $k.$ For example, $S_3$ is the sequence $1,4,7,10,\ldots.$ For how many values of $k$ does $S_k$ contain the term 2005?'
problem='The $9$ members of a baseball team went to an ice-cream parlor after their game. Each player had a single scoop cone of chocolate, vanilla, or strawberry ice cream. At least one player chose each flavor, and the number of players who chose chocolate was greater than the number of players who chose vanilla, which was greater than the number of players who chose strawberry. Let $N$ be the number of different assignments of flavors to players that meet these conditions. Find the remainder when $N$ is divided by $1000.$'
CUDA_VISIBLE_DEVICES=9
python baseline_vllm.py \
    --model_name "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    --problem "$problem" \
    --max_new_tokens 8192 \
    --repeat_time 32 \
    --output_file "results/my_baseline_7b.json"