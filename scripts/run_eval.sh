#!/bin/bash
#SBATCH --job-name=seqr_finetune
#SBATCH --partition=h100,a100
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --output=work_dirs/slurm/precompute/precompute_%j.out
#SBATCH --error=work_dirs/slurm/precompute/precompute_%j.err
#SBATCH --time=48:00:00
# problem='Let $ABCD$ be an isosceles trapezoid with $AD=BC$ and $AB<CD.$ Suppose that the distances from $A$ to the lines $BC,CD,$ and $BD$ are $15,18,$ and $10,$ respectively. Let $K$ be the area of $ABCD.$ Find $\sqrt2 \cdot K.$'

CUDA_VISIBLE_DEVICES=8 python -m src.pipeline.eval_cache \
    --kv_cache_dir ./DS7B_512_128_complex_takeaways_ \
    --follow_up_prompt "$problem" \
    --repeat_time 1



# CUDA_VISIBLE_DEVICES=7 python -m src.pipeline.eval_cache \
#     --kv_cache_dir ./hf_precomputed_kv_budget_680_comp_complex_takeaways \
#     --repeat_time 16 \
#     --follow_up_prompt "Three vertices of a cube are $P=(7,12,10)$ , $Q=(8,8,1)$ , and $R=(11,3,9)$ . What is the surface area of the cube?" \
#     --output_file ./results/results_budget_ 680_comp_complex_takeaways_doc0.json

# CUDA_VISIBLE_DEVICES=6 python -m src.pipeline.eval_cache \
#     --kv_cache_dir ./hf_precomputed_kv_budget_680_comp_complex_notepad \
#     --repeat_time 16 \
#     --follow_up_prompt "Three vertices of a cube are $P=(7,12,10)$ , $Q=(8,8,1)$ , and $R=(11,3,9)$ . What is the surface area of the cube?" \
#     --output_file ./results/results_budget_680_comp_complex_notepad_doc0.json


# python -m src.pipeline.eval_cache \
#     --kv_cache_dir ./DS7B_512_128_complex_takeaways_ \
#     --repeat_time 16 \
#     --follow_up_prompt "In $\\triangle PQR$ , $PR=15$ , $QR=20$ , and $PQ=25$ . Points $A$ and $B$ lie on $\\overline{PQ}$ , points $C$ and $D$ lie on $\\overline{QR}$ , and points $E$ and $F$ lie on $\\overline{PR}$ , with $PA=QB=QC=RD=RE=PF=5$ . Find the area of hexagon $ABCDEF$ ." \
#     --output_file ./results/test.json