import json
import os
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache, 
)
from rkv.config import get_compression_config
from rkv.monkeypatch import replace_llama, replace_qwen2, replace_qwen3
from utils import PatchedDynamicCache


def load_precomputed_kv_as_dynamic_cache(
    kv_path: str,
    doc_id: Optional[int] = None,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
) -> PatchedDynamicCache:
    """
    Load precomputed KV cache from disk and convert to PatchedDyn`amicCache.
    
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

    # Load the dict (can use weights_only=True since we now save pure tensors)
    kv_cache_dict = torch.load(kv_path, map_location="cpu", weights_only=False)
    
    # Check the structure
    if not isinstance(kv_cache_dict, dict):
        raise ValueError(f"Expected dict, got {type(kv_cache_dict)}")
    available_docs = list(kv_cache_dict.keys())
    print(f"Available document IDs: {available_docs}")
    # Select document to load
    if doc_id is None:
        doc_id = available_docs[0]
        print(f"No doc_id specified, loading first document: {doc_id}")
    
    if doc_id not in kv_cache_dict:
        raise ValueError(f"Document {doc_id} not found. Available: {available_docs}")

    doc_cache = kv_cache_dict[doc_id]
    past_key_values = PatchedDynamicCache.from_legacy_cache(tuple(tuple(p.to(device) for p in layer) for layer in doc_cache))
    
    return past_key_values

def load_precomputed_kv_as_dynamic_cache_single(
    kv_path: str,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
) -> PatchedDynamicCache:

    past_key_values = torch.load(kv_path, map_location="cpu", weights_only=False)
    past_key_values = PatchedDynamicCache.from_legacy_cache(tuple(tuple(p.to(device) for p in layer) for layer in past_key_values))
    return past_key_values

def verify_preload_with_dynamic_cache(
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    kv_cache_dir: str = "hf_precomputed_kv",
    follow_up_prompt: str = "Make an overall summary of the documents",
    max_new_tokens: int = 256,
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


    compression_config = get_compression_config()
    compression_config["method"] = "rkv"
    compression_config["method_config"].update({
        "budget": 512,
        "window_size": 256,
        "kernel_size": 7,
        "mix_lambda": 0.07,
        "retain_ratio": 0.66,
        "retain_direction": "last",
        "record_kept_token_indices": False,
    })

    lower_name = model_name.lower()
    if "llama" in lower_name:
        replace_llama(compression_config)
    elif "qwen3" in lower_name:
        replace_qwen3(compression_config)
    elif "qwen" in lower_name:
        replace_qwen2(compression_config)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

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
        # attn_implementation="flash_attention_2",
        attn_implementation="sdpa",
        use_cache=True,
    )
    model.to("cuda:0")
    model.eval()

    for key, value in compression_config.items():
        setattr(model.config, key, value)

    model.config.divide_method = "step_length" # newline is better actually 
    model.config.divide_length = 128
    model.config.compression_content = "all"

    newline_tokens = [
        tokenizer.encode("\n", add_special_tokens=False),
        tokenizer.encode(".\n", add_special_tokens=False),
        tokenizer.encode(")\n", add_special_tokens=False),
        tokenizer.encode("\n\n", add_special_tokens=False),
        tokenizer.encode(".\n\n", add_special_tokens=False),
        tokenizer.encode(")\n\n", add_special_tokens=False),
    ]
    model.newline_token_ids = [seq[-1] for seq in newline_tokens if len(seq) > 0]
    think_token = tokenizer.encode("</think>", add_special_tokens=False)
    model.after_think_token_ids = [think_token[-1]] if len(think_token) > 0 else []

    # Load precomputed KV cache using PatchedDynamicCache
    past_key_values = load_precomputed_kv_as_dynamic_cache(
        kv_path=kv_path,
        doc_id=1,  # If None, will load the first document
        device=device,
        dtype=model.dtype,
    )

    # past_key_values = load_precomputed_kv_as_dynamic_cache_single(
    #     kv_path="./past_key_values_1.pt",
    #     device=device,
    #     dtype=model.dtype,
    # )

    cache_seq_len = past_key_values.get_seq_length()
    print(f"Precomputed context seq_len={cache_seq_len}")
    
    print(f"\nPrecomputed context includes:")
    for i, (doc, summary) in enumerate(zip(documents, summaries), 1):
        print(f"\nDocument {i}:")
        print(f"  Content: {doc[:100]}...")
        print(f"  Summary: {summary[:100]}...")
    print(f"\nFollow-up prompt: {follow_up_prompt}")

    context_token_ids_dict = metadata.get("context_token_ids", {})
    # Extract the token list from the dict (assuming single entry with key "1")
    if isinstance(context_token_ids_dict, dict):
        context_token_list = list(context_token_ids_dict.values())[0]
    else:
        context_token_list = context_token_ids_dict
    # Convert to tensor and add batch dimension to match tokenizer output shape
    context_input_ids = torch.tensor([context_token_list], dtype=torch.long, device=device)
    context_attention_mask = torch.ones_like(context_input_ids, dtype=torch.bool, device=device)

    follow_up_inputs = tokenizer(
        follow_up_prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )
    follow_up_input_ids = follow_up_inputs["input_ids"].to(device)
    follow_up_attention_mask = follow_up_inputs["attention_mask"].to(device)

    input_ids = torch.cat([context_input_ids, follow_up_input_ids], dim=-1)
    attention_mask = torch.cat([context_attention_mask, follow_up_attention_mask], dim=-1)

    print(f"Context token shape: {context_input_ids.shape}")
    print(f"Follow-up token shape: {follow_up_input_ids.shape}")
    print(f"Concatenated input shape: {input_ids.shape}")
    print(f"Past KV cache seq_len: {cache_seq_len}")

    print(f"\nGenerating with past_key_values (max_new_tokens={max_new_tokens})...")

    past_key_values.set_seq_length(context_input_ids.size(1))
    print(f"Set past_key_values seq_len={past_key_values.get_seq_length()}")
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)

    print(f"\n{'='*80}")
    print("生成结果")
    print("="*80)
    print(f"\nFull output:\n{generated_text}")
    print(f"\nResponse only:\n{response}")

    return {
        "generated_text": generated_text,
        "response": response,
        "input_length": input_ids.shape[1],
        "output_length": outputs.shape[1],
        "cache_seq_len": seq_len,
    }



if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "9"


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
        default="hf_precomputed_kv",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Make an overall summary of the documents",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--doc_id",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    verify_preload_with_dynamic_cache(
        model_name=args.model_name,
        kv_cache_dir=args.kv_cache_dir,
        follow_up_prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        doc_id=args.doc_id,
    )
