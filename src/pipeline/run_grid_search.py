#!/usr/bin/env python3
"""
Grid search script for CompressingInContext precomputation and verification.

This script performs a grid search over one scenario:
- Fixed summarization length, varying budget sizes and prompt complexity levels

Usage:
    python run_grid_search.py --budget-range 128 256 512 --complexities simple medium complex
"""

import subprocess
import argparse
import os
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional
import json

def run_precomputation(budget: int, max_new_tokens: int, summary_complexity: str, mode: str, num_epochs: int, data_path: str, cache_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Run precomputation with given budget, max_new_tokens, and summary complexity.

    Args:
        budget: KV cache budget size
        max_new_tokens: Maximum number of tokens to generate per document
        summary_complexity: Level of prompt complexity (simple, medium, complex)

    Returns:
        Path to the generated cache directory
    """
    print(f"\n{'='*80}")
    print(f"Running precomputation: budget={budget}, max_new_tokens={max_new_tokens}, complexity={summary_complexity}, mode={mode}")
    print(f"{'='*80}")

    # Get script directory to resolve relative paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    PRECOMPUTED_DIR_NAME = f"hf_precomputed_kv_budget_{budget}_maxlen_{max_new_tokens}_complexity_{summary_complexity}_mode_{mode}"
    if cache_dir is not None:
        precomputed_path = cache_dir / PRECOMPUTED_DIR_NAME
    else:
        precomputed_path = project_root / PRECOMPUTED_DIR_NAME
    data_path = Path(data_path)
    
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        return None
    
    # Run precomputation - explicitly pass environment to ensure CUDA visibility
    result = subprocess.run(
        [
            'python', '-m', 'src.pipeline.precompute_cache',
            '--data_path', str(data_path),
            '--budget', str(budget),
            '--summary_complexity', summary_complexity,
            '--precomputed_dir', str(precomputed_path),
            '--num_epochs', str(num_epochs),
            '--mode', mode,
        ],
        capture_output=False,
        text=True,
        cwd=str(project_root),  # Run from project root
        env=os.environ.copy()   # Explicitly inherit environment (CUDA_VISIBLE_DEVICES, etc.)
    )

    if result.returncode != 0:
        print(f"WARNING: Precomputation failed for budget={budget}, max_new_tokens={max_new_tokens}, complexity={summary_complexity}, mode={mode}")
        return None

    if not precomputed_path.exists():
        print(f"WARNING: Expected cache directory {precomputed_path} not found")
        return None

    print(f"✓ Precomputation completed: {precomputed_path}")
    return precomputed_path


def run_verification(
    cache_dir: Path,
    model_name: str = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    max_new_tokens: int = 8192,
    output_dir: Path = None
) -> Optional[Path]:
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
        complexity = parts[8] if len(parts) > 8 else None
        complexity_suffix = f"_complexity_{complexity}" if complexity else ""
        output_filename = f"results_budget_{budget}_maxlen_{max_len}{complexity_suffix}.json"
    except (IndexError, ValueError):
        output_filename = f"results_{cache_dir.name}.json"

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        output_dir = project_root / f"results/verification_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / output_filename

    # Get script directory to resolve relative paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    command = [
        'python -m', 'src.pipeline.eval_cache',
        '--model_name', model_name,
        '--kv_cache_dir', str(cache_dir),
        '--max_new_tokens', str(max_new_tokens),
        '--output_file', str(output_file),
        '--repeat_time', '16',
    ]

    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=False, text=True, cwd=str(project_root), env=os.environ.copy())

    if result.returncode != 0:
        print(f"WARNING: Verification failed for {cache_dir}")
        return None

    print(f"✓ Verification completed: {output_file}")
    return output_file


def scenario_budget_and_complexity(
    max_new_tokens: int,
    budget_range: List[int],
    complexity_levels: List[str],
    modes: List[str],
    num_epochs: int,
    data_path: str,
    cache_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    run_verification_after: bool = True,
    verification_max_tokens: int = 8192,
    model_name: str = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
) -> List[Tuple[int, int, str, str, Optional[Path]]]:
    """
    Scenario: Fixed summarization length, varying budget sizes, prompt complexity, and modes.

    Args:
        max_new_tokens: Max new tokens used for all runs
        budget_range: Range of budget values (e.g., [64, 128, 256, ..., 8192])
        complexity_levels: Prompt complexity levels (simple, medium, complex)
        modes: List of modes to search over (takeaways, notepad)
        cache_dir: Directory to store precomputed caches (default: project root)
        results_dir: Directory to store verification results (default: project root/results)
        run_verification_after: Whether to run verification after precomputation
        verification_max_tokens: Max tokens for verification
        model_name: Model used for verification

    Returns:
        List of (budget, max_new_tokens, complexity, mode, cache_dir) tuples
    """
    print("\n" + "="*80)
    print("SCENARIO: Fixed summarization length, varying budget sizes, prompt complexity, and modes")
    print("="*80)

    results = []
    cache_dirs_list = []

    for mode in modes:
        print(f"\n=== Testing mode={mode} ===")
        for complexity in complexity_levels:
            print(f"\n--- Testing complexity={complexity} ---")
            for budget in budget_range:
                precomputed_dir = run_precomputation(budget, max_new_tokens, complexity, mode, num_epochs, data_path, cache_dir=cache_dir)
                results.append((budget, max_new_tokens, complexity, mode, precomputed_dir))
                if precomputed_dir:
                    cache_dirs_list.append(precomputed_dir)

    # Run verification on all generated caches
    if run_verification_after and cache_dirs_list:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        if results_dir is not None:
            output_dir = results_dir / f"scenario_verification_{timestamp}"
        else:
            output_dir = project_root / f"results/scenario_verification_{timestamp}"
        print(f"\n{'='*80}")
        print(f"Running verification ({len(cache_dirs_list)} caches)")
        print(f"{'='*80}")

        for precomputed_dir in cache_dirs_list:
            run_verification(
                precomputed_dir,
                model_name=model_name,
                max_new_tokens=verification_max_tokens,
                output_dir=output_dir,
            )

    return results


def save_grid_search_summary(results: List[Tuple[int, int, str, str, Optional[Path]]], output_file: Path):
    """Save a summary of grid search results."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_runs": len(results),
        "successful_runs": sum(1 for *_, cache_dir in results if cache_dir is not None),
        "failed_runs": sum(1 for *_, cache_dir in results if cache_dir is None),
        "configurations": [
            {
                "budget": budget,
                "max_new_tokens": max_new_tokens,
                "summary_complexity": complexity,
                "mode": mode,
                "cache_dir": str(cache_dir) if cache_dir else None,
                "success": cache_dir is not None
            }
            for budget, max_new_tokens, complexity, mode, cache_dir in results
        ]
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Grid search summary saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Grid search for CompressingInContext parameters (budget x prompt complexity x mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default ranges (all modes)
  python run_grid_search.py

  # Custom budgets, complexities, and modes
  python run_grid_search.py --budget-range 128 256 512 1024 --complexities simple medium --modes takeaways notepad

  # Only test takeaways mode
  python run_grid_search.py --modes takeaways

  # Skip verification (only precompute)
  python run_grid_search.py --no-verify
        """
    )

    # Scenario parameters
    parser.add_argument('--budget-range', '--s2-budget-range', nargs='+', type=int,
                       default=[64 + 160 + 128, 256 + 160 + 128, 512 + 160 + 128, 
                                1024 + 160 + 128, 2048 + 160 + 128, 8192 + 160 + 128],
                       help='Budget range (default: 352 544 800 1312 2336 8480)')
    parser.add_argument('--complexities', '--s2-complexities', nargs='+', type=str,
                       default=["complex",],
                       help='Prompt complexity levels (default: simple medium complex)')
    parser.add_argument('--max-new-tokens', type=int, default=2048,
                       help='Max new tokens for precomputation (applied to all runs)')
    parser.add_argument('--data-path', type=str, default="data_math.json",
                       help='Path to the data file')
    # Verification parameters
    parser.add_argument('--no-verify', action='store_true',
                       help='Skip verification after precomputation')
    parser.add_argument('--verification-max-tokens', type=int, default=8192,
                       help='Max tokens for verification (default: 8192)')
    parser.add_argument('--model-name', type=str,
                       default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
                       help='Model name for verification')
    parser.add_argument('--modes', nargs='+', type=str, default=["takeaways", "notepad"],
                       choices=["takeaways", "notepad"], help="Modes to search over (default: takeaways notepad)")
    parser.add_argument('--num-epochs', type=int, default=1,
                       help='Number of times to repeat documents')
    # Directory parameters
    parser.add_argument('--cache-dir', type=str, default=None,
                       help='Directory to store precomputed KV caches (default: project root)')
    parser.add_argument('--results-dir', type=str, default=None,
                       help='Directory to store verification results (default: project root/results)')
    args = parser.parse_args()

    # Convert directory arguments to Path objects if provided
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    results_dir = Path(args.results_dir) if args.results_dir else None

    results = scenario_budget_and_complexity(
        max_new_tokens=args.max_new_tokens,
        budget_range=args.budget_range,
        complexity_levels=args.complexities,
        run_verification_after=not args.no_verify,
        verification_max_tokens=args.verification_max_tokens,
        model_name=args.model_name,
        modes=args.modes,
        num_epochs=args.num_epochs,
        data_path=args.data_path,
        cache_dir=cache_dir,
        results_dir=results_dir,
    )

    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    if results_dir is not None:
        summary_file = results_dir / f"scenario_summary_{timestamp}.json"
    else:
        summary_file = project_root / f"results/scenario_summary_{timestamp}.json"
    save_grid_search_summary(results, summary_file)

    # Print final summary
    print("\n" + "="*80)
    print("GRID SEARCH COMPLETED")
    print("="*80)
    print(f"Total configurations tested: {len(results)}")
    print(f"Successful: {sum(1 for *_, cache_dir in results if cache_dir is not None)}")
    print(f"Failed: {sum(1 for *_, cache_dir in results if cache_dir is None)}")
    print("="*80)


if __name__ == "__main__":
    main()
