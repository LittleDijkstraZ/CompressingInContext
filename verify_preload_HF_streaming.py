import json
import os
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional


import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache,
    TextIteratorStreamer,
)

from utils import PatchedDynamicCache


def load_precomputed_kv_as_dynamic_cache(
    kv_path: str,
    doc_id: Optional[int] = None,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
) -> PatchedDynamicCache:
    """
    Load precomputed KV cache from disk and convert to PatchedDynamicCache.

    The cache file contains multiple documents in format:
    {
        doc_id: {
            layer_idx: {"key": tensor, "value": tensor}
        }
    }

    Args:
        kv_path: Path to the saved KV cache file (past_key_values_list.pt)
        doc_id: Document ID to load (1-indexed). If None, loads the first document.
        device: Target device for tensors. If None, uses CPU.
        dtype: Target dtype for tensors

    Returns:
        PatchedDynamicCache instance with loaded KV pairs for the specified document
    """
    if device is None:
        device = torch.device("cpu")

    print(f"Loading precomputed KV cache from: {kv_path}")

    kv_cache_dict = torch.load(kv_path, map_location="cpu", weights_only=False)

    if not isinstance(kv_cache_dict, dict):
        raise ValueError(f"Expected dict, got {type(kv_cache_dict)}")
    available_docs = list(kv_cache_dict.keys())
    print(f"Available document IDs: {available_docs}")

    if doc_id is None:
        doc_id = available_docs[0]
        print(f"No doc_id specified, loading first document: {doc_id}")

    if doc_id not in kv_cache_dict:
        raise ValueError(f"Document {doc_id} not found. Available: {available_docs}")

    doc_cache = kv_cache_dict[doc_id]
    past_key_values = PatchedDynamicCache.from_legacy_cache(
        tuple(tuple(p.to(device) for p in layer) for layer in doc_cache)
    )

    return past_key_values


def load_precomputed_kv_as_dynamic_cache_single(
    kv_path: str,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
) -> PatchedDynamicCache:
    past_key_values = torch.load(kv_path, map_location="cpu", weights_only=False)
    past_key_values = PatchedDynamicCache.from_legacy_cache(
        tuple(tuple(p.to(device) for p in layer) for layer in past_key_values)
    )
    return past_key_values


def verify_preload_with_dynamic_cache_streaming(
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    kv_cache_dir: str = "hf_precomputed_kv",
    follow_up_prompt: str = None,
    max_new_tokens: int = 512,
    doc_id: Optional[int] = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata_path = Path(kv_cache_dir) / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    with metadata_path.open("r") as fp:
        metadata = json.load(fp)

    kv_path = metadata["kv_path"]
    seq_len = metadata.get("seq_len")
    documents = metadata.get("documents", [])
    summaries = metadata.get("summaries", [])

    print(f"\nLoaded metadata:")
    print(f"  KV cache path: {kv_path}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Number of documents: {len(documents)}")

    # compression_config = get_compression_config()
    # compression_config["method"] = "rkv"
    # compression_config["method_config"].update(
    #     {
    #         "budget": 512,
    #         "window_size": 256,
    #         "kernel_size": 7,
    #         "mix_lambda": 0.07,
    #         "retain_ratio": 0.66,
    #         "retain_direction": "last",
    #         "record_kept_token_indices": False,
    #     }
    # )

    # lower_name = model_name.lower()
    # if "llama" in lower_name:
    #     replace_llama(compression_config)
    # elif "qwen3" in lower_name:
    #     replace_qwen3(compression_config)
    # elif "qwen" in lower_name:
    #     replace_qwen2(compression_config)
    # else:
    #     raise ValueError(f"Unsupported model: {model_name}")

    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        padding_side="left",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        use_cache=True,
    )
    model.to("cuda:0")
    model.eval()

    # for key, value in compression_config.items():
    #     setattr(model.config, key, value)

    # model.config.divide_method = "step_length"
    # model.config.divide_length = 128
    # model.config.compression_content = "all"

    # newline_tokens = [
    #     tokenizer.encode("\n", add_special_tokens=False),
    #     tokenizer.encode(".\n", add_special_tokens=False),
    #     tokenizer.encode(")\n", add_special_tokens=False),
    #     tokenizer.encode("\n\n", add_special_tokens=False),
    #     tokenizer.encode(".\n\n", add_special_tokens=False),
    #     tokenizer.encode(")\n\n", add_special_tokens=False),
    # ]
    # model.newline_token_ids = [seq[-1] for seq in newline_tokens if len(seq) > 0]
    # think_token = tokenizer.encode("</think>", add_special_tokens=False)
    # model.after_think_token_ids = [think_token[-1]] if len(think_token) > 0 else []

    past_key_values = load_precomputed_kv_as_dynamic_cache(
        kv_path=kv_path,
        doc_id=1 if doc_id is None else doc_id,
        device=device,
        dtype=model.dtype,
    )

    cache_seq_len = past_key_values.get_seq_length()
    print(f"Precomputed context seq_len={cache_seq_len}")

    print(f"\nPrecomputed context includes:")
    for i, (doc, summary) in enumerate(zip(documents, summaries), 1):
        print(f"\nDocument {i}:")
        print(f"  Content: {doc[:100]}...")
        print(f"  Summary: {summary[:100]}...")
    print(f"\nFollow-up prompt: {follow_up_prompt}")

    context_token_ids_dict = metadata.get("context_token_ids", {})
    if isinstance(context_token_ids_dict, dict):
        context_token_list = list(context_token_ids_dict.values())[0]
    else:
        context_token_list = context_token_ids_dict
    context_input_ids = torch.tensor(
        [context_token_list], dtype=torch.long, device=device
    )
    context_attention_mask = torch.ones_like(
        context_input_ids, dtype=torch.bool, device=device
    )

    follow_up_inputs = tokenizer(
        follow_up_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    follow_up_input_ids = follow_up_inputs["input_ids"].to(device)
    follow_up_attention_mask = follow_up_inputs["attention_mask"].to(device)

    input_ids = torch.cat([context_input_ids, follow_up_input_ids], dim=-1)
    attention_mask = torch.cat(
        [context_attention_mask, follow_up_attention_mask], dim=-1
    )
    print(f"\nAttention mask sum: {attention_mask.sum().item()}")

    print(f"Context token shape: {context_input_ids.shape}")
    print(f"Follow-up token shape: {follow_up_input_ids.shape}")
    print(f"Concatenated input shape: {input_ids.shape}")
    print(f"Past KV cache seq_len: {cache_seq_len}")

    print(f"\nStreaming generation with past_key_values (max_new_tokens={max_new_tokens})...")

    past_key_values.set_seq_length(context_input_ids.size(1))
    # past_key_values.set_seq_length(384)

    cache_length = past_key_values.get_seq_length()  # Should be 2214
    print(f"Set past_key_values seq_len={cache_length}")
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=False,
        skip_special_tokens=False,
    )

    generation_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,

        # input_ids=follow_up_input_ids,
        # attention_mask=follow_up_attention_mask,

        max_new_tokens=max_new_tokens,
        temperature=0.6,
        top_p=0.95,
        do_sample=True,
        # pad_token_id=tokenizer.pad_token_id,
        # eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        return_dict_in_generate=True,
        streamer=streamer,
    )

    result_container: Dict[str, Any] = {}

    def _generate():
        with torch.inference_mode():
            result_container["generation"] = model.generate(**generation_kwargs)

    generation_thread = threading.Thread(target=_generate, daemon=True)
    generation_thread.start()

    streamed_response: List[str] = []
    try:
        for token_text in streamer:
            print(token_text, end="", flush=True)
            streamed_response.append(token_text)
    finally:
        generation_thread.join()

    print()  # ensure newline after streaming output

    generation_output = result_container.get("generation")
    if generation_output is None:
        raise RuntimeError("Generation thread did not return outputs.")

    outputs = generation_output.sequences
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    response = tokenizer.decode(
        outputs[0][input_ids.shape[1] :], skip_special_tokens=False
    )

    print(f"\n{'=' * 80}")
    print("生成结果")
    print("=" * 80)
    print(f"\nFull output:\n{generated_text}")
    # print(f"\nResponse only:\n{response}")

    return {
        "generated_text": generated_text,
        "response": response,
        "streamed_chunks": streamed_response,
        "input_length": input_ids.shape[1],
        "output_length": outputs.shape[1],
        "cache_seq_len": seq_len,
    }


def apply_chat_template(input_text, model_name: str) -> str:
    if model_name == "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B":
        eos_token = "<｜end▁of▁sentence｜>"
        bos = "<｜begin▁of▁sentence｜>"
        user_token = "<｜User｜>"
        assistant_token = "<｜Assistant｜>"
        think_token = "<think>"

    else:
        raise ValueError(f"Unsupported model for chat template: {model_name}")

    prompt = (
        "Now, can you solve this new problem using what you've learned, as well as your prior knowledge:\n"
        "here is the problem:\n"
        "###Problem:\n"
    )
    prompt_after = (
        "\nYou are going to reason about this problem within the <think>...</think> tags. And give the solution afterwards."
        "\nPut your final answer within \\boxed{}."
    )

    generation_prompt = (
        f"{eos_token}{bos}{user_token}{prompt}{input_text}{prompt_after}\n{assistant_token}{think_token}"
    )
    return generation_prompt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    )
    parser.add_argument(
        "--kv_cache_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=8192,
    )
    parser.add_argument(
        "--doc_id",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--follow_up_prompt",
        type=str,
        default=None,
        help="The follow-up prompt to use for generation."
    )
    import datetime
    time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument(
        "--output_file",
        type=str,
        default=f'./raw_results_{time_stamp}.json',
        help="File to save the results to."
    )



    args = parser.parse_args()

    if args.follow_up_prompt is None:
        args.follow_up_prompt = (
            "On $\\triangle ABC$ points $A,D,E$, and $B$ lie that order on side $\\overline{AB}$ "
            "with $AD=4, DE=16$, and $EB=8$. Points $A,F,G$, and $C$ lie in that order on side "
            "$\\overline{AC}$ with $AF=13, FG=52$, and $GC=26$. Let $M$ be the reflection of $D$ "
            "through $F$, and let $N$ be the reflection of $G$ through $E$. Quadrilateral $DEGF$ has "
            "area 288. Find the area of heptagon $AFNBCEM$."
        )

    follow_up_prompt = apply_chat_template(args.follow_up_prompt, args.model_name)

    results = verify_preload_with_dynamic_cache_streaming(
        model_name=args.model_name,
        kv_cache_dir=args.kv_cache_dir,
        follow_up_prompt=follow_up_prompt,
        max_new_tokens=args.max_new_tokens,
        doc_id=args.doc_id,
    )

    # Save results
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")
    else:
        # Fallback to old behavior if no output file is specified
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        budget, max_len = None, None
        try:
            parts = args.kv_cache_dir.split('_')
            budget = int(parts[4])
            max_len = int(parts[6])
        except (IndexError, ValueError):
            pass

        if budget is not None and max_len is not None:
            result_filename = f"results_budget_{budget}_maxlen_{max_len}.json"
        else:
            result_filename = "results_default.json"
            
        result_path = results_dir / result_filename
        with result_path.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {result_path}")

