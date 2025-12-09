"""
Baseline Transformers evaluation script.
Runs model inference on a problem without any memory/compression using transformers.
This is the baseline method that does NOT use any precomputed KV cache.
"""
import json
import datetime
from pathlib import Path
from typing import List, Dict, Any
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


def apply_chat_template(input_text: str, model_name: str, tokenizer) -> str:
    """
    Apply chat template for baseline: directly ask the model to solve the problem.
    No think tags, just the problem and requirement to put answer in boxed.
    """
    # Create a simple user message with the problem
    user_message = (
        f"Please solve the following problem:\n"
        f"###Problem:\n"
        f"{input_text}\n"
        f"\nPut your final answer within \\boxed{{}}."
    )
    
    if model_name == "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" or model_name == 'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B':
        eos_token = "<｜end▁of▁sentence｜>"
        bos = "<｜begin▁of▁sentence｜>"
        user_token = "<｜User｜>"
        assistant_token = "<｜Assistant｜>"
        think_token = "<think>"
        end_think_token = "</think>"
        prompt = f"{bos}{user_token}\n{user_message}\n{assistant_token}\n"
    else:
        raise ValueError(f"Unsupported model for chat template: {model_name}")
    
    return prompt


def run_baseline_transformers(
    model_name: str = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    problem: str = None,
    max_new_tokens: int = 8192,
    repeat_time: int = 1,
    temperature: float = 0.6,
    top_p: float = 0.95,
    device: str = None,
) -> Dict[str, Any]:
    """
    Run baseline Transformers inference on a problem WITHOUT any precomputed KV cache.
    This is the baseline method that directly generates without any context compression.

    Args:
        model_name: HuggingFace model name
        problem: The problem text to solve
        max_new_tokens: Maximum number of tokens to generate
        repeat_time: Number of output sequences to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        device: Device to run on (e.g., 'cuda:0', 'cuda:1'). If None, uses 'cuda' if available.

    Returns:
        Dictionary containing generation results (same format as eval_cache.py)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Format the problem using chat template (no think tags, just problem + boxed requirement)
    prompt = apply_chat_template(problem, model_name, tokenizer)
    print(f"\nFormatted prompt:\n{prompt[:500]}...")

    # Get stop token IDs
    eos_token_ids = []
    if tokenizer.eos_token_id is not None:
        eos_token_ids.append(tokenizer.eos_token_id)

    # Add additional stop tokens for DeepSeek-R1
    additional_stop_tokens = ["<｜end▁of▁sentence｜>", "</think>"]
    for stop_token in additional_stop_tokens:
        token_id = tokenizer.convert_tokens_to_ids(stop_token)
        if token_id != tokenizer.unk_token_id and token_id not in eos_token_ids:
            eos_token_ids.append(token_id)

    print(f"EOS token IDs: {eos_token_ids}")

    print(f"\nLoading model: {model_name}")
    print(f"Using device: {device}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        use_cache=True,
    )
    model.to(device)
    model.eval()

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]

    print(f"\nGenerating {repeat_time} sample(s) with max_new_tokens={max_new_tokens}...")
    print("NOTE: This is the BASELINE method - no precomputed KV cache is used.")

    # Generate multiple sequences
    all_results: List[Dict[str, Any]] = []
    
    with torch.inference_mode():
        for idx in range(repeat_time):
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_token_ids if eos_token_ids else tokenizer.eos_token_id,
                use_cache=True,
            )
            
            # Decode the generated sequence
            generated_ids = outputs[0]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Extract only the new tokens (response)
            response_ids = generated_ids[input_length:]
            response = tokenizer.decode(response_ids, skip_special_tokens=True)
            
            output_length = generated_ids.shape[0]

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
        "streamed_chunks": [],  # No streaming in offline mode
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Baseline Transformers evaluation (no KV cache)")
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
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g., 'cuda:0', 'cuda:1'). If not specified, uses 'cuda' if available.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="File to save the results to. If not provided, results/baseline_transformers_*.json will be used.",
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

    results = run_baseline_transformers(
        model_name=args.model_name,
        problem=args.problem,
        max_new_tokens=args.max_new_tokens,
        repeat_time=args.repeat_time,
        device=args.device,
    )

    # Save results
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)
        time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"baseline_transformers_{time_stamp}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

