#!/usr/bin/env python3
"""
Extract answers from batch generation results and calculate pass@k.

Supports answers inside \\boxed{...} and plain mentions such as
"answer is 754" or "final answer: **754**". Computes pass@k as
correct / total_tryouts for each JSON file, and reports whether the
extracted answer was boxed.
"""

import json
import re
from pathlib import Path
from typing import Tuple, Optional


def extract_answer(text: str) -> Tuple[Optional[str], bool]:
    """
    Extract an answer string and whether it came from a boxed form.
    Priority:
      1) \\boxed{...}
      2) Patterns like "answer is 754" or "final answer: **754**"
      3) Fallback to the last standalone number in the text
    """
    if not text:
        return None, False

    # 1) Boxed answers
    boxed_matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed_matches:
        answer = boxed_matches[-1].strip()
        return (answer if answer else None), True

    # 2) Plain "answer is ..." variants
    plain_patterns = [
        r"(?i)(?:final\s+answer|the\s+answer|answer)\s*(?:is|=|:)?\s*\*{0,2}\s*([-+]?\d+(?:\.\d+)?)",
    ]
    for pat in plain_patterns:
        matches = re.findall(pat, text)
        if matches:
            answer = matches[-1].strip().strip("*").strip()
            return (answer if answer else None), False

    # 3) Fallback: last numeric token in the text (avoids missing unflagged answers)
    number_matches = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if number_matches:
        return number_matches[-1], False

    return None, False


def check_correct_answer(text: str, correct_answer: str = "588") -> bool:
    """
    Check if the text contains the correct answer (boxed or unboxed).
    """
    answer, _ = extract_answer(text)
    if answer is None:
        return False
    return str(answer) == str(correct_answer)


def process_results_file(filepath: Path, correct_answer: str = "588"):
    """
    Process a single results JSON file with batch generations.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        all_generations = data.get("all_generations", [])

        if not all_generations:
            return {
                "file": filepath.name,
                "pass_at_k": 0.0,
                "k": 0,
                "correct_count": 0,
                "total_tryouts": 0,
                "boxed_count": 0,
                "individual_results": [],
                "error": "No generations found",
            }

        correct_count = 0
        boxed_count = 0
        individual_results = []

        for idx, generation in enumerate(all_generations):
            response = generation.get("response", "")
            extracted_answer, is_boxed = extract_answer(response)
            is_correct = (
                extracted_answer is not None
                and str(extracted_answer) == str(correct_answer)
            )

            if is_correct:
                correct_count += 1
            if is_boxed and extracted_answer is not None:
                boxed_count += 1

            individual_results.append(
                {
                    "tryout": idx + 1,
                    "extracted_answer": extracted_answer,
                    "is_correct": is_correct,
                    "is_boxed": is_boxed,
                }
            )

        total_tryouts = len(all_generations)
        pass_at_k = pass_at_k_metric(total_tryouts, correct_count, k=total_tryouts)

        return {
            "file": filepath.name,
            "pass_at_k": pass_at_k,
            "k": total_tryouts,
            "correct_count": correct_count,
            "total_tryouts": total_tryouts,
            "boxed_count": boxed_count,
            "individual_results": individual_results,
        }

    except Exception as e:
        return {
            "file": filepath.name,
            "pass_at_k": 0.0,
            "k": 0,
            "correct_count": 0,
            "total_tryouts": 0,
            "boxed_count": 0,
            "individual_results": [],
            "error": str(e),
        }


def pass_at_k_metric(n: int, c: int, k: int) -> float:
    """
    Compute pass@k as a simple proportion of correct rollouts:
      pass@k = c / n where n is total rollouts and c is correct rollouts.
    """
    if n == 0:
        return 0.0
    return c / n


def main():
    """
    Main function to process all result files in the results directory.
    """
    results_dir = Path("results/scenario_verification_20251129_050944")

    if not results_dir.exists():
        print(f"Error: Results directory '{results_dir}' not found!")
        return

    json_files = sorted(results_dir.rglob("*.json"))

    if not json_files:
        print(f"No JSON files found in '{results_dir}'")
        return

    print(f"Found {len(json_files)} JSON files to process\n")
    print("=" * 80)

    all_results = [process_results_file(json_file) for json_file in json_files]

    # Sort results by pass@k descending
    all_results.sort(key=lambda r: r.get("pass_at_k", 0.0), reverse=True)

    for result in all_results:
        print(f"File: {result['file']}")
        k = result.get("k", result.get("total_tryouts", 0))
        print(
            f"  Pass@{k}: {result['pass_at_k']:.2%} "
            f"({result['correct_count']}/{result['total_tryouts']} correct)"
        )
        print(
            f"  Boxed answers: {result.get('boxed_count', 0)}/"
            f"{result['total_tryouts']}"
        )
        if "error" in result:
            print(f"  Error: {result['error']}")
        else:
            print("  Individual tryouts:")
            for ind_result in result["individual_results"]:
                status = "✓" if ind_result["is_correct"] else "✗"
                answer = (
                    ind_result["extracted_answer"]
                    if ind_result["extracted_answer"]
                    else "None"
                )
                boxed_flag = " [boxed]" if ind_result.get("is_boxed") else ""
                print(
                    f"    Tryout {ind_result['tryout']}: "
                    f"{status} (answer: {answer}{boxed_flag})"
                )
        print("-" * 80)

    output_file = "extracted_answers_passk.csv"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("filename,k_used,pass_at_k,correct_count,total_tryouts,boxed_count\n")
        for result in all_results:
            k = result.get("k", result.get("total_tryouts", 0))
            f.write(
                f"{result['file']},{k},{result['pass_at_k']:.4f},"
                f"{result['correct_count']},{result['total_tryouts']},"
                f"{result.get('boxed_count', 0)}\n"
            )

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
