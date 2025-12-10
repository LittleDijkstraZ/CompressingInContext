"""
Baseline vLLM evaluation script.
Runs model inference on a problem without any memory/compression.
This is the baseline method that does NOT use any precomputed KV cache.
"""
import json
import datetime
from pathlib import Path
from typing import List, Dict, Any

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def apply_chat_template(input_text: str, model_name: str, tokenizer) -> str:
    """
    Apply chat template for baseline: directly ask the model to solve the problem.
    Uses standard messages format with "Let's think step by step" instruction.
    """
    # Create messages in the format: [{"role": "user", "content": question + instruction}]
    messages = [
        {
            "role": "user",
            "content": input_text + "\nLet's think step by step and output the final answer within \\boxed{}."
        }
    ]
    
    # Use tokenizer's chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    return prompt


def run_baseline_vllm(
    model_name: str = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    problem: str = None,
    max_new_tokens: int = 8192,
    repeat_time: int = 1,
    temperature: float = 0.6,
    top_p: float = 0.95,
) -> Dict[str, Any]:
    """
    Run baseline vLLM inference on a problem WITHOUT any precomputed KV cache.
    This is the baseline method that directly generates without any context compression.

    Args:
        model_name: HuggingFace model name
        problem: The problem text to solve
        max_new_tokens: Maximum number of tokens to generate
        repeat_time: Number of output sequences to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter

    Returns:
        Dictionary containing generation results (same format as eval_cache.py)
    """
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # Format the problem using chat template (no think tags, just problem + boxed requirement)
    prompt = apply_chat_template(problem, model_name, tokenizer)
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
        max_tokens=max_new_tokens,
        n=repeat_time,
        stop_token_ids=stop_token_ids if stop_token_ids else None,
    )

    print(f"\nLoading model: {model_name}")
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="bfloat16",
    )

    print(f"\nGenerating {repeat_time} sample(s) with max_new_tokens={max_new_tokens}...")
    print("NOTE: This is the BASELINE method - no precomputed KV cache is used.")
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
        print(f"Generate {idx + 1}/{repeat_time}")
        print("=" * 80)
        print(f"\nFull output:\n{generated_text}")

        all_results.append({
            "generated_text": generated_text,
            "response": response,
            "input_length": input_length,
            "output_length": output_length,
            "cache_seq_len": None,  # No cache for baseline
        })

    return {
        "all_generations": all_results,
        "num_repeats": repeat_time,
        "streamed_chunks": [],  # No streaming in vLLM offline mode
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Baseline vLLM evaluation (no KV cache)")
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=8192,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--repeat_time",
        type=int,
        default=1,
        help="Number of times to repeat the generation.",
    )
    parser.add_argument(
        "--problem",
        type=str,
        default=None,
        help="The problem to solve.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="File to save the results to. If not provided, results/baseline_*.json will be used.",
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
        # args.problem = ""

    results = run_baseline_vllm(
        model_name=args.model_name,
        problem=args.problem,
        max_new_tokens=args.max_new_tokens,
        repeat_time=args.repeat_time,
    )

    # Save results
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)
        time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"baseline_{time_stamp}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

