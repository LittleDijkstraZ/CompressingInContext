#!/usr/bin/env python3
"""
Generate randomized grid search configurations.

This script generates all combinations of hyperparameters, randomly shuffles them,
and saves the first N configurations to a JSON file for use by grid_search_random.sh.

Usage:
    python generate_random_configs.py --num-configs 50 --output configs.json
    python generate_random_configs.py --num-configs 100 --seed 42
"""

import argparse
import json
import random
from itertools import product
from pathlib import Path
from datetime import datetime


def generate_configs(
    model_names: list,
    budgets: list,
    window_sizes: list,
    rotations: list,
    data_limits: list,
    modes: list,
    num_epochs: list,
) -> list:
    """Generate all valid combinations of hyperparameters.

    Filters out illegal combinations based on:
    - budget must be > window_size + 80
    """
    configs = []
    filtered_count = 0
    for model_name, budget, window_size, rotation, data_limit, mode, epochs in product(
        model_names, budgets, window_sizes, rotations, data_limits, modes, num_epochs
    ):
        # Filter: budget must be greater than window_size + 80
        if budget <= window_size + 80:
            filtered_count += 1
            continue

        configs.append({
            "model_name": model_name,
            "budget": budget,
            "window_size": window_size,
            "rotation": rotation,
            "data_limit": data_limit,
            "mode": mode,
            "num_epochs": epochs,
        })

    if filtered_count > 0:
        print(f"Filtered out {filtered_count} illegal combinations (budget <= window_size + 80)")

    return configs


def main():
    parser = argparse.ArgumentParser(
        description="Generate randomized grid search configurations"
    )
    
    # Number of configs to generate
    parser.add_argument('--num-configs', '-n', type=int, default=50,
                       help='Number of configurations to select (default: 50, use -1 for all)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (default: None)')
    parser.add_argument('--output', '-o', type=str, default='random_configs.json',
                       help='Output file path (default: random_configs.json)')
    
    # Hyperparameter ranges
    parser.add_argument('--model-names', nargs='+', type=str,
                       default=['deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', 'PlanePaper/LEAD-7B'],
                       help='Model names')
    parser.add_argument('--budgets', nargs='+', type=int,
                       default=[256, 512, 1024, 2048, 4096, 8192],
                       help='Budget values')
    parser.add_argument('--window-sizes', nargs='+', type=int,
                       default=[64, 128, 256],
                       help='Window size values')
    parser.add_argument('--rotations', nargs='+', type=str,
                       default=['none', '16', '256', '1024', '2048', '4096', '8192'],
                       help='Rotation values (use "none" for no rotation)')
    parser.add_argument('--data-limits', nargs='+', type=int,
                       default=[1, 2, 3, 5, 8, 13],
                       help='Data limit values')
    parser.add_argument('--modes', nargs='+', type=str,
                       default=['takeaways', 'notepad', 'none'],
                       help='Mode values')
    parser.add_argument('--num-epochs', nargs='+', type=int,
                       default=[1, 3, 5],
                       help='Number of epochs values')

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")

    # Generate all configurations
    all_configs = generate_configs(
        model_names=args.model_names,
        budgets=args.budgets,
        window_sizes=args.window_sizes,
        rotations=args.rotations,
        data_limits=args.data_limits,
        modes=args.modes,
        num_epochs=args.num_epochs,
    )
    
    print(f"Total possible configurations: {len(all_configs)}")
    
    # Shuffle configurations
    random.shuffle(all_configs)
    
    # Select first N configurations
    if args.num_configs == -1:
        selected_configs = all_configs
    else:
        selected_configs = all_configs[:args.num_configs]
    
    print(f"Selected configurations: {len(selected_configs)}")
    
    # Prepare output
    output_data = {
        "generated_at": datetime.now().isoformat(),
        "seed": args.seed,
        "total_possible": len(all_configs),
        "num_selected": len(selected_configs),
        "hyperparameter_ranges": {
            "model_names": args.model_names,
            "budgets": args.budgets,
            "window_sizes": args.window_sizes,
            "rotations": args.rotations,
            "data_limits": args.data_limits,
            "modes": args.modes,
            "num_epochs": args.num_epochs,
        },
        "configs": selected_configs,
    }
    
    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Configurations saved to: {output_path}")
    
    # Print sample configs
    print("\nSample configurations (first 5):")
    for i, config in enumerate(selected_configs[:5]):
        print(f"  {i+1}. {config}")


if __name__ == "__main__":
    main()

