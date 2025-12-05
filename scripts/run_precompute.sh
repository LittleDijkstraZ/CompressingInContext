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
export RKV_DEBUG_COMPRESSION=1
# python src/pipeline/precompute_cache.py --num_epochs 1 --data_path ./limo_clustering_results/k16/clusters/cluster_6.json
# python -m src.pipeline.precompute_cache --num_epochs 1 --data_path ./limo_clustering_results/k40/clusters/cluster_31.json 2>&1 | tee precompute.log

python -m src.pipeline.precompute_cache --num_epochs 1 --data_path ./data.json 2>&1 | tee precompute.log