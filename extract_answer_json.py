#!/usr/bin/env python3
"""
Extract answers from batch generation results and calculate pass@8.

For each results file with 8 repeat tryouts, checks how many tryouts contain
\boxed{588} in the response field. Calculates pass@8 = correct_count / 8.
"""

import json
import re
from pathlib import Path


def extract_boxed_answer(text):
    """
    Extract the answer from text that contains \\boxed{answer}.
    Returns the answer if found, otherwise returns None.

    Args:
        text: String containing the generated text

    Returns:
        The extracted answer or None if not found
    """
    if not text:
        return None

    # Pattern to match \boxed{...}
    pattern = r'\\boxed\{([^}]+)\}'

    matches = re.findall(pattern, text)

    if matches:
        # Return the last match (in case there are multiple)
        answer = matches[-1].strip()
        return answer if answer else None
    else:
        return None


def check_correct_answer(text, correct_answer="588"):
    """
    Check if the text contains the correct answer in \\boxed{} format.

    Args:
        text: String containing the generated text
        correct_answer: The correct answer to check against (default: "588")

    Returns:
        True if correct answer found, False otherwise
    """
    answer = extract_boxed_answer(text)
    if answer is None:
        return False
    return str(answer) == str(correct_answer)


def process_results_file(filepath, correct_answer="588"):
    """
    Process a single results JSON file with batch generations.

    Args:
        filepath: Path to the JSON file
        correct_answer: The correct answer to check against (default: "588")

    Returns:
        Dictionary with filename, pass@8, correct_count, total_tryouts, and individual results
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Get all generations
        all_generations = data.get('all_generations', [])
        num_repeats = data.get('num_repeats', len(all_generations))

        if not all_generations:
            return {
                'file': filepath.name,
                'pass_at_8': 0.0,
                'correct_count': 0,
                'total_tryouts': 0,
                'individual_results': [],
                'error': 'No generations found'
            }

        # Check each tryout
        correct_count = 0
        individual_results = []

        for idx, generation in enumerate(all_generations):
            response = generation.get('response', '')
            is_correct = check_correct_answer(response, correct_answer)
            
            if is_correct:
                correct_count += 1
            
            extracted_answer = extract_boxed_answer(response)
            individual_results.append({
                'tryout': idx + 1,
                'extracted_answer': extracted_answer,
                'is_correct': is_correct
            })

        # Calculate pass@8: proportion of correct answers out of all tryouts
        total_tryouts = len(all_generations)
        pass_at_8 = correct_count / total_tryouts if total_tryouts > 0 else 0.0

        return {
            'file': filepath.name,
            'pass_at_8': pass_at_8,
            'correct_count': correct_count,
            'total_tryouts': total_tryouts,
            'individual_results': individual_results
        }

    except Exception as e:
        return {
            'file': filepath.name,
            'pass_at_8': 0.0,
            'correct_count': 0,
            'total_tryouts': 0,
            'individual_results': [],
            'error': str(e)
        }


def main():
    """
    Main function to process all result files in the results directory.
    """
    results_dir = Path('results/scenario1_verification_20251129_001337')

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

    for json_file in json_files:
        result = process_results_file(json_file)
        all_results.append(result)

        # Print result
        print(f"File: {result['file']}")
        print(f"  Pass@8: {result['pass_at_8']:.2%} ({result['correct_count']}/{result['total_tryouts']} correct)")
        if 'error' in result:
            print(f"  Error: {result['error']}")
        else:
            print(f"  Individual tryouts:")
            for ind_result in result['individual_results']:
                status = "✓" if ind_result['is_correct'] else "✗"
                answer = ind_result['extracted_answer'] if ind_result['extracted_answer'] else "None"
                print(f"    Tryout {ind_result['tryout']}: {status} (answer: {answer})")
        print("-" * 80)

    # Save detailed results to CSV
    output_file = 'extracted_answers_pass8.csv'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("filename,pass_at_8,correct_count,total_tryouts\n")
        for result in all_results:
            f.write(f"{result['file']},{result['pass_at_8']:.4f},{result['correct_count']},{result['total_tryouts']}\n")

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()

