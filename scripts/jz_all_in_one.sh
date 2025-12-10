#!/bin/bash
#SBATCH --job-name=seqr_finetune
#SBATCH --partition=h100,a100
#SBATCH --gpus=1
#SBATCH --exclude=c007,h02
#SBATCH --mem=80G
#SBATCH --output=work_dirs/slurm/precompute/precompute_%j.out
#SBATCH --error=work_dirs/slurm/precompute/precompute_%j.err
#SBATCH --time=48:00:00

# python src/pipeline/precompute_cache.py --num_epochs 1 --data_path ./limo_clustering_results/k16/clusters/cluster_6.json

# --data_path ./src/clustering/limo_clustering_results/k40/clusters/cluster_22.json --mode takeaways --recompute \

export RKV_DEBUG_COMPRESSION=1

python -m src.pipeline.precompute_cache_none --num_epochs 1 \
    --model_name "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    --data_path ./src/clustering/similar_questions_result_aime25_3.json \
    --mode takeaways \
    --rotate true \
    --budget 256 \
    --data_limit 1 \
    --target_rotation_position 256 \
    --window_size 128 \
    --divide_method step_length \
    --recompute \
    2>&1 | tee precompute_jz.log

problem='The $9$ members of a baseball team went to an ice-cream parlor after their game. Each player had a single scoop cone of chocolate, vanilla, or strawberry ice cream. At least one player chose each flavor, and the number of players who chose chocolate was greater than the number of players who chose vanilla, which was greater than the number of players who chose strawberry. Let $N$ be the number of different assignments of flavors to players that meet these conditions. Find the remainder when $N$ is divided by $1000.$'
python -m src.pipeline.eval_cache \
    --kv_cache_dir ???
    --repeat_time 16 \
    --follow_up_prompt "$problem" 
