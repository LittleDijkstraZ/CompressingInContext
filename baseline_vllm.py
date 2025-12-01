"""
Baseline vLLM evaluation script.
Runs model inference on a problem without any memory/compression.
"""
import json
import datetime
from pathlib import Path
from typing import List, Dict, Any

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def run_baseline_vllm(
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    problem: str = None,
    max_tokens: int = 8192,
    n: int = 1,
    temperature: float = 0.6,
    top_p: float = 0.95,
) -> Dict[str, Any]:
    """
    Run baseline vLLM inference on a problem.

    Args:
        model_name: HuggingFace model name
        problem: The problem text to solve
        max_tokens: Maximum number of tokens to generate
        n: Number of output sequences to generate (equivalent to repeat_time)
        temperature: Sampling temperature
        top_p: Top-p sampling parameter

    Returns:
        Dictionary containing generation results
    """
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # Format the problem using the tokenizer's chat template
    messages = [{"role": "user", "content": problem}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    print(f"\nFormatted prompt:\n{prompt[:500]}...")

    # Get stop token IDs
    stop_token_ids = []
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)

    # Add additional stop tokens for DeepSeek-R1
    additional_stop_tokens = ["<｜end▁of▁sentence｜>", "</think>"]
    for stop_token in additional_stop_tokens:
        token_id = tokenizer.convert_tokens_to_ids(stop_token)
        if token_id != tokenizer.unk_token_id and token_id not in stop_token_ids:
            stop_token_ids.append(token_id)

    print(f"Stop token IDs: {stop_token_ids}")

    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=n,
        stop_token_ids=stop_token_ids if stop_token_ids else None,
    )

    print(f"\nLoading model: {model_name}")
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="bfloat16",
    )

    print(f"\nGenerating {n} sample(s) with max_tokens={max_tokens}...")
    outputs = llm.generate([prompt], sampling_params)

    # Process outputs
    all_results: List[Dict[str, Any]] = []
    request_output = outputs[0]

    input_length = len(request_output.prompt_token_ids)

    for idx, output in enumerate(request_output.outputs):
        generated_text = prompt + output.text
        response = output.text
        output_length = input_length + len(output.token_ids)

        print(f"\n{'=' * 80}")
        print(f"Generation {idx + 1}/{n}")
        print("=" * 80)
        print(f"\nResponse:\n{response[:1000]}...")

        all_results.append({
            "generated_text": generated_text,
            "response": response,
            "input_length": input_length,
            "output_length": output_length,
            "cache_seq_len": None,  # No cache for baseline
        })

    return {
        "all_generations": all_results,
        "num_repeats": n,
        "streamed_chunks": [],  # No streaming in vLLM offline mode
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Baseline vLLM evaluation")
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8192,
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of output sequences to generate (equivalent to repeat_time).",
    )
    parser.add_argument(
        "--problem",
        type=str,
        default=None,
        help="The problem to solve.",
    )
    time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="File to save the results to. Defaults to results/baseline_{timestamp}.json",
    )

    args = parser.parse_args()

    if args.problem is None:
        args.problem = (
            "On $\\triangle ABC$ points $A,D,E$, and $B$ lie that order on side $\\overline{AB}$ "
            "with $AD=4, DE=16$, and $EB=8$. Points $A,F,G$, and $C$ lie in that order on side "
            "$\\overline{AC}$ with $AF=13, FG=52$, and $GC=26$. Let $M$ be the reflection of $D$ "
            "through $F$, and let $N$ be the reflection of $G$ through $E$. Quadrilateral $DEGF$ has "
            "area 288. Find the area of heptagon $AFNBCEM$."
        )

    results = run_baseline_vllm(
        model_name=args.model_name,
        problem=args.problem,
        max_tokens=args.max_tokens,
        n=args.n,
    )

    # Save results
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / f"baseline_{time_stamp}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

