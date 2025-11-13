
import json
import os
from transformers.cache_utils import DynamicCache
import random
import shutil
from pathlib import Path
from contextlib import ExitStack
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    DynamicCache,
)

from rkv.config import get_compression_config
from rkv.monkeypatch import replace_llama, replace_qwen2, replace_qwen3




class TqdmProgress(StoppingCriteria):
    def __init__(self, total_steps: int):
        self.total = total_steps
        self.pbar = tqdm(total=total_steps, desc="Generating", leave=False)
        self._init_len = None

    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> bool:
        cur_len = input_ids.shape[-1]
        if self._init_len is None:
            self._init_len = cur_len
            return False

        added = cur_len - (self._init_len + self.pbar.n)
        if added > 0:
            self.pbar.update(min(added, self.total - self.pbar.n))
        return False

    def close(self):
        self.pbar.close()


def build_document_prompt(doc_id: int, document: str, ) -> str:
    
    instruction = (
        "Summarize the current document in 1 sentence."
    )
    
    parts: List[str] = []
    parts.append(instruction)
    parts.extend([
        f"```document {doc_id}```",
        document,
        "```",
    ])
    return "\n".join(parts)



def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _load_tokenizer_and_model() -> tuple[AutoTokenizer, AutoModelForCausalLM, torch.device]:
    lower_name = HF_MODEL_ID.lower()
    if "llama" in lower_name:
        replace_llama(compression_config)
    elif "qwen3" in lower_name:
        replace_qwen3(compression_config)
    elif "qwen" in lower_name:
        replace_qwen2(compression_config)
    else:
        raise ValueError(f"Unsupported model for R-KV patch: {HF_MODEL_ID}")

    tokenizer = AutoTokenizer.from_pretrained(
        HF_MODEL_ID,
        use_fast=True,
        padding_side="left",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=ATTN_IMPL,
        use_cache=True,
        low_cpu_mem_usage=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    for key, value in compression_config.items():
        setattr(model.config, key, value)

    newline_candidates = ["\n", ".\n", ")\n", "\n\n", ".\n\n", ")\n\n"]
    newline_token_ids: List[int] = []
    for pattern in newline_candidates:
        ids = tokenizer.encode(pattern, add_special_tokens=False)
        if ids:
            newline_token_ids.append(ids[-1])
    model.newline_token_ids = newline_token_ids

    think_ids = tokenizer.encode("</think>", add_special_tokens=False)
    model.after_think_token_ids = [think_ids[-1]] if think_ids else []

    HF_GENERATION_KWARGS.setdefault("pad_token_id", tokenizer.pad_token_id)
    HF_GENERATION_KWARGS.setdefault("eos_token_id", tokenizer.eos_token_id)

    return tokenizer, model, device






def _iter_cache_layers(past_key_values: Any):
    cache_object = past_key_values.to_legacy_cache() if hasattr(past_key_values, "to_legacy_cache") else past_key_values
    if cache_object is None:
        raise RuntimeError("Model.generate did not return usable past_key_values.")

    if isinstance(cache_object, dict):
        items = list(cache_object.items())
        for fallback_idx, (raw_layer_idx, payload) in enumerate(items):
            yield _normalize_layer_idx(raw_layer_idx, fallback_idx), payload
        return

    if isinstance(cache_object, (list, tuple)):
        for layer_idx, payload in enumerate(cache_object):
            yield layer_idx, payload
        return

    raise TypeError(f"Unsupported past_key_values container: {type(cache_object).__name__}")


def _normalize_layer_idx(raw_idx: Any, fallback: int) -> int:
    if isinstance(raw_idx, int):
        return raw_idx
    if isinstance(raw_idx, str):
        digits = ''.join(ch for ch in raw_idx if ch.isdigit())
        if digits:
            return int(digits)
    return fallback


def _tensor_candidates(candidate: Any):
    todo = [candidate]
    seen = set()

    while todo:
        current = todo.pop()
        if current is None:
            continue
        current_id = id(current)
        if current_id in seen:
            continue
        seen.add(current_id)

        if isinstance(current, torch.Tensor):
            yield current
            continue

        if isinstance(current, dict):
            todo.extend(current.values())
            todo.extend(current.get(name) for name in ("key", "value", "keys", "values", "k", "v") if name in current)
            continue

        if isinstance(current, (list, tuple)):
            todo.extend(current)
            continue

        for name in ("key", "keys", "value", "values", "k", "v", "key_cache", "value_cache"):
            if hasattr(current, name):
                todo.append(getattr(current, name))


def _fallback_sources(full_cache: Any, layer_idx: Optional[int]):
    if full_cache is None or layer_idx is None:
        return

    for attr in ("key_cache", "value_cache"):
        store = getattr(full_cache, attr, None)
        if isinstance(store, (list, tuple)) and 0 <= layer_idx < len(store):
            yield store[layer_idx]
        elif isinstance(store, dict):
            yield store.get(layer_idx) or store.get(str(layer_idx))

    layers = getattr(full_cache, "layers", None)
    if isinstance(layers, (list, tuple)) and 0 <= layer_idx < len(layers):
        yield layers[layer_idx]


def _materialize_layer_cache(layer_payload: Any, *, full_cache: Optional[Any] = None, layer_idx: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
    tensors = []
    for tensor in _tensor_candidates(layer_payload):
        tensors.append(tensor)
        if len(tensors) == 2:
            break

    if len(tensors) < 2:
        for extra_source in _fallback_sources(full_cache, layer_idx):
            for tensor in _tensor_candidates(extra_source):
                tensors.append(tensor)
                if len(tensors) == 2:
                    break
            if len(tensors) == 2:
                break

    if len(tensors) < 2:
        raise RuntimeError(f"Unable to materialize tensor KV cache for layer {layer_idx}.")

    return tensors[0], tensors[1]


def _collect_layer_tensors(past_key_values: Any) -> Dict[int, Dict[str, torch.Tensor]]:
    collected: Dict[int, Dict[str, torch.Tensor]] = {}
    for layer_idx, payload in _iter_cache_layers(past_key_values):
        key_tensor, value_tensor = _materialize_layer_cache(payload, full_cache=past_key_values, layer_idx=layer_idx)
        collected[layer_idx] = {
            "key": key_tensor.detach().cpu(),
            "value": value_tensor.detach().cpu(),
        }
    return collected


def _append_context(context_prefix: str, doc_id: int, document: str, summary: str) -> str:
    parts: List[str] = []
    if context_prefix:
        parts.append(context_prefix)
    parts.extend([f"Document {doc_id}:", document])
    summary = summary.strip()
    if summary:
        parts.extend(["Summary:", summary])
    return "\n".join(part for part in parts if part).strip()



def apply_chat_template(input_text, model_name: str) -> str:
    if model_name == 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B':
        bos = "<｜begin▁of▁sentence｜>"
        user_token = "<｜User｜>"
        assistant_token = "<｜Assistant｜>"
    
    else:
        raise ValueError(f"Unsupported model for chat template: {model_name}")
    
    prompt = (
        "You are trying learning to solve a certain type of math problems from some examples. "
        "Given the problem, reasoning, and solution, you will try to learn how to solve such problems on your own, "
        "writing down the key takeaways that can help you solve similar problems in the future. "
    )
    generation_prompt = f"""{bos}{user_token}{prompt}
{assistant_token}{input_text}"""
    return generation_prompt




def compute_dynamic_cache(documents: List[str]) -> Dict[str, Any]:
    metadata_path = PRECOMPUTED_DIR / "metadata.json"
    if metadata_path.exists() and not os.getenv("RKV_RECOMPUTE_KV"):
        with metadata_path.open("r") as fp:
            return json.load(fp)

    tokenizer = TOKENIZER
    model = MODEL
    device = DEVICE
    summaries: List[str] = []
    context_prefix = ""
    past_key_values_dict: Dict[int, DynamicCache] = {}
    token_records: Dict[int, List[int]] = {}
    past_conversation = []
    past_key_values =  DynamicCache()
    past_context = ''

    for doc_id, example in enumerate(tqdm(documents, desc="Precomputing HF KV")):
        
        past_context += example + "\n###Takeaways:\n"
        generation_prompt = apply_chat_template(
            input_text = past_context,
            model_name = HF_MODEL_ID,
        )

        print("=="*100)
        print("History: \n", generation_prompt)
        print("=="*100)


        inputs = tokenizer(generation_prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[-1]
    
        with ExitStack() as stack:
            progress = TqdmProgress(total_steps=HF_MAX_NEW_TOKENS)
            stack.callback(progress.close)
            with torch.inference_mode():
                generation = model.generate(
                    **inputs,
                    **HF_GENERATION_KWARGS,
                    use_cache=True,
                    return_dict_in_generate=True,
                    stopping_criteria=StoppingCriteriaList([progress]),
                    past_key_values=past_key_values,
                )

                past_key_values = generation.past_key_values

        # assume only one sequence
        full_text_ids = generation.sequences[0, :].detach().cpu()

        # truncate last token if it's eos_token. else, just leave it as a broken sentence.
        last_token = full_text_ids[-1]
        if last_token == tokenizer.eos_token_id:
            full_text_ids = full_text_ids[:-1]
            for layer_idx in range(len(past_key_values.key_cache)):
                past_key_values.key_cache[layer_idx] = past_key_values.key_cache[layer_idx][:, :, :-1, :]
                past_key_values.value_cache[layer_idx] = past_key_values.value_cache[layer_idx][:, :, :-1, :]

        summary = tokenizer.decode(full_text_ids[input_len:], skip_special_tokens=True).strip()
        raw = tokenizer.decode(full_text_ids[input_len:], skip_special_tokens=False)
        print(f"raw length={len(raw)} vs summary length={len(summary)}")
        print(f"Warning: raw!=summary") if raw != summary else None
        print("=" * 100)
        print(f"Generated summary: {summary}")
        print(f"Generated summary raw: {raw}")
        print("=" * 100)

        print(f"cache_length= {past_key_values.get_seq_length()} vs full_text_ids_tokenized_len= {len(full_text_ids)}")

        past_context += raw + "\n" # add a line switch at the end. the model will also encode this


    # Get sequence length from first cache
    cache_len = past_key_values.get_seq_length() 
    print(f"cached seq_len: {cache_len}")
    print(f"total_input_tokens: {input_len}")

    past_key_values_dict[1] = past_key_values
    
    kv_path = PRECOMPUTED_DIR / "past_key_values_dict.pt"
    torch.save(past_key_values_dict, kv_path)
    print(f"Saved {len(past_key_values_dict)} document caches to {kv_path}")

    context_text = past_context
    print("text to cache: ", context_text)
    context_token_ids = tokenizer.encode(context_text)
    print(f"Total context tokens to cache: {len(context_token_ids)}")
    print(f"cache length=", cache_len)

    metadata = {
        "kv_path": str(kv_path.resolve()),
        "seq_len": cache_len,
        "documents": documents,
        "summaries": summaries,
        "context_token_ids": context_token_ids,
    }
    
    # Save metadata to file (will overwrite existing metadata.json)
    with metadata_path.open("w") as fp:
        json.dump(metadata, fp, indent=2)
    
    print(f"Saved metadata to {metadata_path}")
    return metadata


if __name__ == "__main__":
   
    # os.environ["CUDA_VISIBLE_DEVICES"] = "9"


    HF_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    # ATTN_IMPL = "flash_attention_2"
    ATTN_IMPL = "sdpa"
    HF_MAX_NEW_TOKENS = 64
    HF_TEMPERATURE = 0.6
    HF_TOP_P = 0.95
    HF_MAX_COMPLETIONS_PER_CALL = 1

    DEFAULT_METHOD = "rkv"
    METHOD_CONFIG = {
        "budget": 384,
        "window_size": 128, # BUG: IDK why budget need to be > window_size here
        "kernel_size": 7,
        "mix_lambda": 0.1,
        "retain_ratio": 0.75,
        "retain_direction": "last",
        "record_kept_token_indices": False,
    }

    HF_GENERATION_KWARGS = {
        "max_new_tokens": HF_MAX_NEW_TOKENS,
        "temperature": HF_TEMPERATURE,
        "top_p": HF_TOP_P,
        "do_sample": True,
        "num_return_sequences": 1,
        "output_scores": False,
        "output_attentions": False,
        "output_hidden_states": False,
    }

    PRECOMPUTED_DIR = Path("hf_precomputed_kv")
    PRECOMPUTED_DIR.mkdir(exist_ok=True)

    compression_config = get_compression_config()
    compression_config["method"] = DEFAULT_METHOD
    compression_config["method_config"].update(METHOD_CONFIG)

    # os.environ["CUDA_VISIBLE_DEVICES"] = "9"
    set_seed()
    TOKENIZER, MODEL, DEVICE = _load_tokenizer_and_model()
    
    import json 
    with open('data.json', 'r') as f:
        data = json.load(f)
    documents = []
    for item in data:
        sample = f"""##Problem ID: {item['id']}
###Problem:\n{item['problem']}\n
###Reasoning:\n{item['reasoning']}\n
###Solution:\n{item['solution']}\n"""
        documents.append(sample.strip())


    # precomputed_metadata = compute_precomputed_kv(documents)
    dynamic_cache_metadata = compute_dynamic_cache(documents)