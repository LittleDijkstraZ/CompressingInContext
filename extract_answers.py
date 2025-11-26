#!/usr/bin/env python3
"""
Extract answers from generation results in the results directory.
Answers should be in 'boxed{}' format, otherwise return -1.
Computes pass@1 (correctness) and generation length for each config.
"""

import json
import re
import os
from pathlib import Path
from collections import defaultdict


def extract_boxed_answer(text):
    """
    Extract the answer from text that contains \\boxed{answer}.
    Returns the answer if found, otherwise returns -1.

    Args:
        text: String containing the generated text

    Returns:
        The extracted answer or -1 if not found
    """
    if not text:
        return -1

    # Pattern to match \boxed{...}
    # This handles nested braces by finding the last occurrence
    pattern = r'\\boxed\{([^}]+)\}'

    matches = re.findall(pattern, text)

    if matches:
        # Return the last match (in case there are multiple)
        answer = matches[-1].strip()
        # Return -1 if the answer is empty
        return answer if answer else -1
    else:
        return -1


def get_generation_length(text):
    """
    Calculate the length of the generated text.

    Args:
        text: String containing the generated text

    Returns:
        Length of the text in characters
    """
    return len(text) if text else 0


def parse_config_from_filename(filename):
    """
    Parse budget and maxlen from filename.
    Expected format: results_budget_XXX_maxlen_YYY.json

    Args:
        filename: Name of the file

    Returns:
        Tuple of (budget, maxlen) or (None, None) if parsing fails
    """
    pattern = r'results_budget_(\d+)_maxlen_(\d+)\.json'
    match = re.match(pattern, filename)

    if match:
        budget = int(match.group(1))
        maxlen = int(match.group(2))
        return budget, maxlen
    return None, None


def process_results_file(filepath, correct_answer="588"):
    """
    Process a single results JSON file and extract metrics.

    Args:
        filepath: Path to the JSON file
        correct_answer: The correct answer to check against (default: "588")

    Returns:
        Dictionary with filename, answer, correctness, and generation length
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract the generated text or response field
        text = data.get('generated_text', '') or data.get('response', '')

        answer = extract_boxed_answer(text)
        gen_length = get_generation_length(text)

        # Check if answer is correct
        is_correct = (str(answer) == str(correct_answer))

        return {
            'file': os.path.basename(filepath),
            'answer': answer,
            'is_correct': is_correct,
            'generation_length': gen_length
        }
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return {
            'file': os.path.basename(filepath),
            'answer': -1,
            'is_correct': False,
            'generation_length': 0,
            'error': str(e)
        }


def main():
    """
    Main function to process all result files in the results directory.
    """
    results_dir = Path('results')

    if not results_dir.exists():
        print(f"Error: Results directory '{results_dir}' not found!")
        return

    # Find all JSON files recursively in the results directory
    json_files = sorted(results_dir.rglob('*.json'))

    if not json_files:
        print(f"No JSON files found in '{results_dir}'")
        return

    print(f"Found {len(json_files)} JSON files to process\n")
    print("=" * 80)

    all_results = []
    config_stats = defaultdict(lambda: {
        'total': 0,
        'correct': 0,
        'total_length': 0,
        'answers': []
    })

    for json_file in json_files:
        result = process_results_file(json_file)
        all_results.append(result)

        # Extract config from filename
        budget, maxlen = parse_config_from_filename(result['file'])

        if budget is not None and maxlen is not None:
            config_key = f"budget_{budget}_maxlen_{maxlen}"
            config_stats[config_key]['total'] += 1
            if result['is_correct']:
                config_stats[config_key]['correct'] += 1
            config_stats[config_key]['total_length'] += result['generation_length']
            config_stats[config_key]['answers'].append(result['answer'])

        # Print result
        print(f"File: {result['file']}")
        print(f"Answer: {result['answer']}")
        print(f"Correct: {result['is_correct']}")
        print(f"Generation Length: {result['generation_length']}")
        if 'error' in result:
            print(f"Error: {result['error']}")
        print("-" * 80)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY BY CONFIG")
    print("=" * 80)

    for config_key in sorted(config_stats.keys()):
        stats = config_stats[config_key]
        pass_at_1 = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        avg_length = stats['total_length'] / stats['total'] if stats['total'] > 0 else 0

        print(f"\nConfig: {config_key}")
        print(f"  Total samples: {stats['total']}")
        print(f"  Correct answers: {stats['correct']}")
        print(f"  Pass@1: {pass_at_1:.2%}")
        print(f"  Average generation length: {avg_length:.0f} characters")

    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)

    total_correct = sum(1 for r in all_results if r['is_correct'])
    total_files = len(all_results)
    overall_pass_at_1 = total_correct / total_files if total_files > 0 else 0

    print(f"Total files processed: {total_files}")
    print(f"Total correct answers: {total_correct}")
    print(f"Overall Pass@1: {overall_pass_at_1:.2%}")

    # Save detailed results to CSV
    output_file = 'extracted_answers.csv'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("filename,answer,is_correct,generation_length\n")
        for result in all_results:
            f.write(f"{result['file']},{result['answer']},{result['is_correct']},{result['generation_length']}\n")

    print(f"\nDetailed results saved to: {output_file}")

    # Save aggregated results by config
    agg_output_file = 'aggregated_results.csv'
    with open(agg_output_file, 'w', encoding='utf-8') as f:
        f.write("config,total_samples,correct_answers,pass_at_1,avg_generation_length\n")
        for config_key in sorted(config_stats.keys()):
            stats = config_stats[config_key]
            pass_at_1 = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            avg_length = stats['total_length'] / stats['total'] if stats['total'] > 0 else 0
            f.write(f"{config_key},{stats['total']},{stats['correct']},{pass_at_1:.4f},{avg_length:.0f}\n")

    print(f"Aggregated results saved to: {agg_output_file}")


if __name__ == '__main__':
    main()

