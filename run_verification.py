import os
import subprocess
import argparse
from datetime import datetime
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run multiple rounds of verification using streaming.")
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=1,
        help="Number of verification rounds to run."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        help="Name of the model to use.",
    )
    parser.add_argument(
        "--cache_parent_dir",
        type=str,
        default=".",
        help="Parent directory containing all the cache directories.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=8192,
        help="Maximum number of new tokens to generate.",
    )

    args = parser.parse_args()

    for i in range(args.num_rounds):
        round_num = i + 1
        print(f"--- Starting Verification Round {round_num}/{args.num_rounds} ---")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/verification_round_{round_num}_{timestamp}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        cache_dirs = [d for d in Path(args.cache_parent_dir).iterdir() if d.is_dir() and "hf_precomputed_kv" in d.name]

        for cache_dir in cache_dirs:
            print(f"  Verifying cache: {cache_dir.name}")
            
            # Extract budget and max_len from cache_dir name to use in the output filename
            try:
                parts = cache_dir.name.split('_')
                budget = parts[4]
                max_len = parts[6]
                output_filename = f"results_budget_{budget}_maxlen_{max_len}.json"
            except (IndexError, ValueError):
                output_filename = f"results_{cache_dir.name}.json"
                
            output_file = output_dir / output_filename
            
            command = [
                'python', './verify_preload_HF_streaming.py',
                '--model_name', args.model_name,
                '--kv_cache_dir', str(cache_dir),
                '--max_new_tokens', str(args.max_new_tokens),
                '--output_file', str(output_file),
            ]
            
            print(f"    Running command: {' '.join(command)}")
            subprocess.run(command, check=True)
        
        print(f"--- Finished Verification Round {round_num}/{args.num_rounds} ---")

    print("All verification rounds are complete.")

if __name__ == "__main__":
    main()
