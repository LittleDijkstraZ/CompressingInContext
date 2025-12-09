#!/usr/bin/env python3
"""
Grid search script for precompute_cache_none.py.

This script performs a grid search over the following parameters:
- model_name: Model to use
- budget: KV cache budget sizes
- data_limit: Number of documents to process
- mode: Precomputation mode (takeaways, notepad, none)
- rotate: Whether to rotate keys
- target_rotation_position: Target rotation position

Usage:
    python run_grid_search_none.py --budget-range 512 1024 --modes none --data-limits 5
"""

import subprocess
import argparse
import os
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional
import json


def run_evaluation(
    model_name: str,
    kv_cache_dir: Path,
    max_new_tokens: int = 8192,
    repeat_time: int = 16,
    eval_results_dir: Optional[Path] = None,
) -> bool:
    """
    Run evaluation on a precomputed KV cache.

    Returns:
        True if evaluation succeeded, False otherwise
    """
    print(f"\n{'='*80}", flush=True)
    print(f"Running evaluation: model={model_name}, cache_dir={kv_cache_dir}", flush=True)
    print(f"  max_new_tokens={max_new_tokens}, repeat_time={repeat_time}", flush=True)
    print(f"{'='*80}", flush=True)

    # Get script directory to resolve relative paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    # Build command
    cmd = [
        'python', '-m', 'src.pipeline.eval_cache',
        '--model_name', model_name,
        '--kv_cache_dir', str(kv_cache_dir),
        '--max_new_tokens', str(max_new_tokens),
        '--repeat_time', str(repeat_time),
    ]

    # Add output file if eval_results_dir is specified
    if eval_results_dir is not None:
        cache_dir_name = kv_cache_dir.name
        output_file = eval_results_dir / f"eval_{cache_dir_name}.json"
        cmd.extend(['--output_file', str(output_file)])
        print(f"  Evaluation results will be saved to: {output_file}", flush=True)

    print(f"Running: {' '.join(cmd)}", flush=True)

    # Run evaluation - use unbuffered output
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    result = subprocess.run(
        cmd,
        capture_output=False,
        text=True,
        cwd=str(project_root),
        env=env,
    )

    if result.returncode != 0:
        print(f"WARNING: Evaluation failed for cache_dir={kv_cache_dir}")
        return False

    print(f"✓ Evaluation completed: {kv_cache_dir}")
    return True


def run_precomputation(
    model_name: str,
    budget: int,
    data_limit: int,
    mode: str,
    rotate: bool,
    target_rotation_position: int,
    data_path: str,
    summary_complexity: str,
    window_size: int,
    num_epochs: int,
    cache_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Run precomputation with given parameters.

    Returns:
        Path to the generated cache directory
    """
    print(f"\n{'='*80}", flush=True)
    print(f"Running precomputation: model={model_name}, budget={budget}, data_limit={data_limit}, "
          f"mode={mode}, rotate={rotate}, target_pos={target_rotation_position}", flush=True)
    print(f"{'='*80}", flush=True)

    # Get script directory to resolve relative paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    rotate_str = "rotate_" if rotate else ""
    PRECOMPUTED_DIR_NAME = (
        f"hf_precomputed_kv_budget_{budget}_window_{window_size}_comp_{summary_complexity}"
        f"_{mode}_{rotate_str}data_{data_limit}"
    )
    if cache_dir is not None:
        precomputed_path = cache_dir / PRECOMPUTED_DIR_NAME
    else:
        precomputed_path = project_root / PRECOMPUTED_DIR_NAME

    data_path = Path(data_path)
    if not data_path.is_absolute():
        data_path = project_root / data_path

    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        return None

    # Build command
    cmd = [
        'python', '-m', 'src.pipeline.precompute_cache_none',
        '--data_path', str(data_path),
        '--model_name', model_name,
        '--budget', str(budget),
        '--data_limit', str(data_limit),
        '--mode', mode,
        '--rotate', str(rotate),
        '--target_rotation_position', str(target_rotation_position),
        '--summary_complexity', summary_complexity,
        '--num_epochs', str(num_epochs),
        '--window_size', str(window_size),
        '--precomputed_dir', str(precomputed_path),
    ]

    print(f"Running: {' '.join(cmd)}", flush=True)

    # Run precomputation - use unbuffered output
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    result = subprocess.run(
        cmd,
        capture_output=False,
        text=True,
        cwd=str(project_root),
        env=env,
    )

    if result.returncode != 0:
        print(f"WARNING: Precomputation failed for budget={budget}, mode={mode}, rotate={rotate}")
        return None

    if not precomputed_path.exists():
        print(f"WARNING: Expected cache directory {precomputed_path} not found")
        return None

    print(f"✓ Precomputation completed: {precomputed_path}")
    return precomputed_path


def run_grid_search(
    model_names: List[str],
    budget_range: List[int],
    data_limits: List[int],
    modes: List[str],
    rotation_configs: List[str],
    data_path: str,
    summary_complexity: str,
    window_size: int,
    num_epochs: int,
    cache_dir: Optional[Path] = None,
    max_new_tokens: int = 8192,
    repeat_time: int = 16,
    eval_results_dir: Optional[Path] = None,
) -> List[Tuple]:
    """
    Run grid search over all parameter combinations.

    rotation_configs: List of "none", "256", "1024", "3072"
        - "none" means rotate=False
        - number means rotate=True with that target_rotation_position
    """
    print("\n" + "="*80)
    print("GRID SEARCH: precompute_cache_none.py")
    print("="*80)

    results = []

    for model_name in model_names:
        print(f"\n=== Model: {model_name} ===")
        for mode in modes:
            print(f"\n--- Mode: {mode} ---")
            for data_limit in data_limits:
                for budget in budget_range:
                    for rot_config in rotation_configs:
                        if rot_config == "none":
                            rotate = False
                            target_pos = 3072  # default, ignored when rotate=False
                        else:
                            rotate = True
                            target_pos = int(rot_config)

                        precomputed_dir = run_precomputation(
                            model_name=model_name,
                            budget=budget,
                            data_limit=data_limit,
                            mode=mode,
                            rotate=rotate,
                            target_rotation_position=target_pos,
                            data_path=data_path,
                            summary_complexity=summary_complexity,
                            window_size=window_size,
                            num_epochs=num_epochs,
                            cache_dir=cache_dir,
                        )

                        # Run evaluation if precomputation succeeded
                        eval_success = False
                        if precomputed_dir is not None:
                            eval_success = run_evaluation(
                                model_name=model_name,
                                kv_cache_dir=precomputed_dir,
                                max_new_tokens=max_new_tokens,
                                repeat_time=repeat_time,
                                eval_results_dir=eval_results_dir,
                            )

                        results.append((
                            model_name, budget, data_limit, mode,
                            rotate, target_pos, precomputed_dir, eval_success
                        ))

    return results


def save_grid_search_summary(results: List[Tuple], output_file: Path):
    """Save a summary of grid search results."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_runs": len(results),
        "successful_precompute": sum(1 for *_, cache_dir, _ in results if cache_dir is not None),
        "failed_precompute": sum(1 for *_, cache_dir, _ in results if cache_dir is None),
        "successful_eval": sum(1 for *_, eval_success in results if eval_success),
        "failed_eval": sum(1 for *_, cache_dir, eval_success in results if cache_dir is not None and not eval_success),
        "configurations": [
            {
                "model_name": model_name,
                "budget": budget,
                "data_limit": data_limit,
                "mode": mode,
                "rotate": rotate,
                "target_rotation_position": target_pos,
                "cache_dir": str(cache_dir) if cache_dir else None,
                "precompute_success": cache_dir is not None,
                "eval_success": eval_success
            }
            for model_name, budget, data_limit, mode, rotate, target_pos, cache_dir, eval_success in results
        ]
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Grid search summary saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Grid search for precompute_cache_none.py parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default parameters (all 4 rotation configs)
  python run_grid_search_none.py --data-path data_math.json

  # Custom budgets and modes
  python run_grid_search_none.py --budget-range 512 1024 2048 --modes none takeaways

  # Only test no rotation
  python run_grid_search_none.py --rotation-configs none

  # Only test rotation with specific positions
  python run_grid_search_none.py --rotation-configs 256 1024

  # Specific model
  python run_grid_search_none.py --model-names PlanePaper/LEAD-7B
        """
    )

    # Model parameters
    parser.add_argument('--model-names', nargs='+', type=str,
                       default=["deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "PlanePaper/LEAD-7B"],
                       choices=["deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "PlanePaper/LEAD-7B"],
                       help='Models to test')

    # Grid search parameters
    parser.add_argument('--budget-range', nargs='+', type=int,
                       default=[256, 512, 1024, 2048, 4096],
                       help='Budget range (default: 256 512 1024 2048 4096)')
    parser.add_argument('--data-limits', nargs='+', type=int,
                       default=[5, 20],
                       help='Data limits (default: 5 20)')
    parser.add_argument('--modes', nargs='+', type=str,
                       default=["takeaways", "notepad", "none"],
                       choices=["takeaways", "notepad", "none"],
                       help='Modes to search over (default: takeaways notepad none)')
    # Rotation configs: "none" means no rotation, numbers mean rotate=True with that position
    parser.add_argument('--rotation-configs', nargs='+', type=str,
                       default=["none", "256", "1024", "3072"],
                       choices=["none", "256", "1024", "3072"],
                       help='Rotation configurations: "none" for no rotation, or position value for rotate=True (default: none 256 1024 3072)')

    # Data and output parameters
    parser.add_argument('--data-path', type=str, default="data_math.json",
                       help='Path to the data file')
    parser.add_argument('--summary-complexity', type=str, default="complex",
                       help='Summary complexity level')
    parser.add_argument('--window-size', type=int, default=128,
                       help='Window size for compression')
    parser.add_argument('--num-epochs', type=int, default=1,
                       help='Number of times to repeat documents')

    # Directory parameters
    parser.add_argument('--cache-dir', type=str, default=None,
                       help='Directory to store precomputed KV caches')
    parser.add_argument('--results-dir', type=str, default=None,
                       help='Directory to store grid search summary')
    parser.add_argument('--eval-results-dir', type=str, default=None,
                       help='Directory to store evaluation results (default: same as --results-dir)')

    # Evaluation parameters
    parser.add_argument('--max-new-tokens', type=int, default=8192,
                       help='Maximum new tokens for evaluation (default: 8192)')
    parser.add_argument('--repeat-time', type=int, default=16,
                       help='Number of times to repeat generation for evaluation (default: 16)')

    args = parser.parse_args()

    # Convert directory arguments to Path objects if provided
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    results_dir = Path(args.results_dir) if args.results_dir else None
    eval_results_dir = Path(args.eval_results_dir) if args.eval_results_dir else results_dir

    # Create eval_results_dir if specified
    if eval_results_dir is not None:
        eval_results_dir.mkdir(parents=True, exist_ok=True)

    results = run_grid_search(
        model_names=args.model_names,
        budget_range=args.budget_range,
        data_limits=args.data_limits,
        modes=args.modes,
        rotation_configs=args.rotation_configs,
        data_path=args.data_path,
        summary_complexity=args.summary_complexity,
        window_size=args.window_size,
        num_epochs=args.num_epochs,
        cache_dir=cache_dir,
        max_new_tokens=args.max_new_tokens,
        repeat_time=args.repeat_time,
        eval_results_dir=eval_results_dir,
    )

    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    if results_dir is not None:
        summary_file = results_dir / f"grid_search_none_summary_{timestamp}.json"
    else:
        summary_file = project_root / f"results/grid_search_none_summary_{timestamp}.json"
    save_grid_search_summary(results, summary_file)

    # Print final summary
    print("\n" + "="*80)
    print("GRID SEARCH COMPLETED")
    print("="*80)
    print(f"Total configurations tested: {len(results)}")
    print(f"Successful precomputation: {sum(1 for *_, cache_dir, _ in results if cache_dir is not None)}")
    print(f"Failed precomputation: {sum(1 for *_, cache_dir, _ in results if cache_dir is None)}")
    print(f"Successful evaluation: {sum(1 for *_, eval_success in results if eval_success)}")
    print(f"Failed evaluation: {sum(1 for *_, cache_dir, eval_success in results if cache_dir is not None and not eval_success)}")
    print("="*80)


if __name__ == "__main__":
    main()

