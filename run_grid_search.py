#!/usr/bin/env python3
"""
Grid search script for CompressingInContext precomputation and verification.

This script performs a grid search over two scenarios:
1. Fixed budget sizes, varying summarization lengths (max_new_tokens)
2. Fixed summarization lengths, varying budget sizes

Usage:
    python run_grid_search.py [--scenario1] [--scenario2] [--both]
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '8'
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import json


def run_precomputation(budget: int, max_new_tokens: int) -> Path:
    """
    Run precomputation with given budget and max_new_tokens.

    Args:
        budget: KV cache budget size
        max_new_tokens: Maximum number of tokens to generate per document

    Returns:
        Path to the generated cache directory
    """
    print(f"\n{'='*80}")
    print(f"Running precomputation: budget={budget}, max_new_tokens={max_new_tokens}")
    print(f"{'='*80}")

    env = os.environ.copy()
    env['HF_MAX_NEW_TOKENS'] = str(max_new_tokens)
    env['budget'] = str(budget)

    # Run precomputation
    result = subprocess.run(
        ['python', './precompute_cache.py'],
        env=env,
        capture_output=False,
        text=True
    )

    if result.returncode != 0:
        print(f"WARNING: Precomputation failed for budget={budget}, max_new_tokens={max_new_tokens}")
        return None

    # Expected cache directory name
    cache_dir = Path(f"hf_precomputed_kv_budget_{budget}_maxlen_{max_new_tokens}")

    if not cache_dir.exists():
        print(f"WARNING: Expected cache directory {cache_dir} not found")
        return None

    print(f"✓ Precomputation completed: {cache_dir}")
    return cache_dir


def run_verification(
    cache_dir: Path,
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    max_new_tokens: int = 8192,
    output_dir: Path = None
) -> Path:
    """
    Run verification on a precomputed cache.

    Args:
        cache_dir: Path to the precomputed cache directory
        model_name: Model to use for verification
        max_new_tokens: Maximum number of tokens to generate during verification
        output_dir: Directory to save verification results

    Returns:
        Path to the output file
    """
    if cache_dir is None or not cache_dir.exists():
        print(f"WARNING: Cache directory {cache_dir} does not exist, skipping verification")
        return None

    print(f"\n{'='*80}")
    print(f"Running verification: {cache_dir.name}")
    print(f"{'='*80}")

    # Extract budget and max_len from cache_dir name
    try:
        parts = cache_dir.name.split('_')
        budget = parts[4]
        max_len = parts[6]
        output_filename = f"results_budget_{budget}_maxlen_{max_len}.json"
    except (IndexError, ValueError):
        output_filename = f"results_{cache_dir.name}.json"

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/verification_{timestamp}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / output_filename

    command = [
        'python', './verify_preload_HF_streaming.py',
        '--model_name', model_name,
        '--kv_cache_dir', str(cache_dir),
        '--max_new_tokens', str(max_new_tokens),
        '--output_file', str(output_file),
    ]

    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"WARNING: Verification failed for {cache_dir}")
        return None

    print(f"✓ Verification completed: {output_file}")
    return output_file


def scenario_1_fixed_budget_varying_length(
    budget_sizes: List[int],
    max_new_tokens_range: List[int],
    run_verification_after: bool = True,
    verification_max_tokens: int = 8192
) -> List[Tuple[int, int, Path]]:
    """
    Scenario 1: Fixed budget sizes, varying summarization lengths.

    Args:
        budget_sizes: List of 2-3 budget sizes to test
        max_new_tokens_range: Range of max_new_tokens values (e.g., [64, 128, 256, ..., 2048])
        run_verification_after: Whether to run verification after precomputation
        verification_max_tokens: Max tokens for verification

    Returns:
        List of (budget, max_new_tokens, cache_dir) tuples
    """
    print("\n" + "="*80)
    print("SCENARIO 1: Fixed budget sizes, varying summarization lengths")
    print("="*80)

    results = []
    cache_dirs = []

    for budget in budget_sizes:
        print(f"\n--- Testing budget={budget} ---")
        for max_new_tokens in max_new_tokens_range:
            cache_dir = run_precomputation(budget, max_new_tokens)
            results.append((budget, max_new_tokens, cache_dir))
            if cache_dir:
                cache_dirs.append(cache_dir)

    # Run verification on all generated caches
    if run_verification_after and cache_dirs:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/scenario1_verification_{timestamp}")
        print(f"\n{'='*80}")
        print(f"Running verification for Scenario 1 ({len(cache_dirs)} caches)")
        print(f"{'='*80}")

        for cache_dir in cache_dirs:
            run_verification(cache_dir, max_new_tokens=verification_max_tokens, output_dir=output_dir)

    return results


def scenario_2_fixed_length_varying_budget(
    max_new_tokens_sizes: List[int],
    budget_range: List[int],
    run_verification_after: bool = True,
    verification_max_tokens: int = 8192
) -> List[Tuple[int, int, Path]]:
    """
    Scenario 2: Fixed summarization lengths, varying budget sizes.

    Args:
        max_new_tokens_sizes: List of 2-3 max_new_tokens values to test
        budget_range: Range of budget values (e.g., [64, 128, 256, ..., 8192])
        run_verification_after: Whether to run verification after precomputation
        verification_max_tokens: Max tokens for verification

    Returns:
        List of (budget, max_new_tokens, cache_dir) tuples
    """
    print("\n" + "="*80)
    print("SCENARIO 2: Fixed summarization lengths, varying budget sizes")
    print("="*80)

    results = []
    cache_dirs = []

    for max_new_tokens in max_new_tokens_sizes:
        print(f"\n--- Testing max_new_tokens={max_new_tokens} ---")
        for budget in budget_range:
            cache_dir = run_precomputation(budget, max_new_tokens)
            results.append((budget, max_new_tokens, cache_dir))
            if cache_dir:
                cache_dirs.append(cache_dir)

    # Run verification on all generated caches
    if run_verification_after and cache_dirs:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/scenario2_verification_{timestamp}")
        print(f"\n{'='*80}")
        print(f"Running verification for Scenario 2 ({len(cache_dirs)} caches)")
        print(f"{'='*80}")

        for cache_dir in cache_dirs:
            run_verification(cache_dir, max_new_tokens=verification_max_tokens, output_dir=output_dir)

    return results


def save_grid_search_summary(results: List[Tuple[int, int, Path]], output_file: Path):
    """Save a summary of grid search results."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_runs": len(results),
        "successful_runs": sum(1 for _, _, cache_dir in results if cache_dir is not None),
        "failed_runs": sum(1 for _, _, cache_dir in results if cache_dir is None),
        "configurations": [
            {
                "budget": budget,
                "max_new_tokens": max_new_tokens,
                "cache_dir": str(cache_dir) if cache_dir else None,
                "success": cache_dir is not None
            }
            for budget, max_new_tokens, cache_dir in results
        ]
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Grid search summary saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Grid search for CompressingInContext parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run both scenarios with default settings
  python run_grid_search.py --both

  # Run only scenario 1
  python run_grid_search.py --scenario1

  # Run only scenario 2 with custom ranges
  python run_grid_search.py --scenario2 --budget-range 128 256 512 1024

  # Skip verification (only precompute)
  python run_grid_search.py --scenario1 --no-verify
        """
    )

    # Scenario selection
    parser.add_argument('--scenario1', action='store_true',
                       help='Run scenario 1: fixed budget, varying summarization length')
    parser.add_argument('--scenario2', action='store_true',
                       help='Run scenario 2: fixed summarization length, varying budget')
    parser.add_argument('--both', action='store_true',
                       help='Run both scenarios')

    # Scenario 1 parameters
    parser.add_argument('--s1-budgets', nargs='+', type=int, default=[192, 384, 768],
                       help='Budget sizes for scenario 1 (default: 192 384 768)')
    parser.add_argument('--s1-length-range', nargs='+', type=int,
                       default=[64, 128, 256, 512, 1024, 2048],
                       help='Max new tokens range for scenario 1 (default: 64 128 256 512 1024 2048)')

    # Scenario 2 parameters
    parser.add_argument('--s2-lengths', nargs='+', type=int, default=[256, 512, 1024],
                       help='Max new tokens sizes for scenario 2 (default: 256 512 1024)')
    parser.add_argument('--s2-budget-range', nargs='+', type=int,
                       default=[64, 128, 256, 512, 1024, 2048, 4096, 8192],
                       help='Budget range for scenario 2 (default: 64 128 256 512 1024 2048 4096 8192)')

    # Verification parameters
    parser.add_argument('--no-verify', action='store_true',
                       help='Skip verification after precomputation')
    parser.add_argument('--verification-max-tokens', type=int, default=8192,
                       help='Max tokens for verification (default: 8192)')
    parser.add_argument('--model-name', type=str,
                       default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                       help='Model name for verification')

    args = parser.parse_args()

    # Determine which scenarios to run
    run_s1 = args.scenario1 or args.both
    run_s2 = args.scenario2 or args.both

    if not (run_s1 or run_s2):
        parser.print_help()
        print("\nError: Must specify at least one scenario (--scenario1, --scenario2, or --both)")
        return

    all_results = []

    # Run scenario 1
    if run_s1:
        results = scenario_1_fixed_budget_varying_length(
            budget_sizes=args.s1_budgets,
            max_new_tokens_range=args.s1_length_range,
            run_verification_after=not args.no_verify,
            verification_max_tokens=args.verification_max_tokens
        )
        all_results.extend(results)

        # Save scenario 1 summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = Path(f"results/scenario1_summary_{timestamp}.json")
        save_grid_search_summary(results, summary_file)

    # Run scenario 2
    if run_s2:
        results = scenario_2_fixed_length_varying_budget(
            max_new_tokens_sizes=args.s2_lengths,
            budget_range=args.s2_budget_range,
            run_verification_after=not args.no_verify,
            verification_max_tokens=args.verification_max_tokens
        )
        all_results.extend(results)

        # Save scenario 2 summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = Path(f"results/scenario2_summary_{timestamp}.json")
        save_grid_search_summary(results, summary_file)

    # Print final summary
    print("\n" + "="*80)
    print("GRID SEARCH COMPLETED")
    print("="*80)
    print(f"Total configurations tested: {len(all_results)}")
    print(f"Successful: {sum(1 for _, _, cache_dir in all_results if cache_dir is not None)}")
    print(f"Failed: {sum(1 for _, _, cache_dir in all_results if cache_dir is None)}")
    print("="*80)


if __name__ == "__main__":
    main()
