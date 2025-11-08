
import json
import os
from transformers.cache_utils import DynamicCache
os.environ["CUDA_VISIBLE_DEVICES"] = "9"
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

HF_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
ATTN_IMPL = "flash_attention_2"
HF_MAX_NEW_TOKENS = 1
HF_TEMPERATURE = 0.6
HF_TOP_P = 0.95
HF_MAX_COMPLETIONS_PER_CALL = 1

DEFAULT_METHOD = "rkv"
METHOD_CONFIG = {
    "budget": 32,
    "window_size": 16,
    "kernel_size": 7,
    "mix_lambda": 0.07,
    "retain_ratio": 0.1,
    "retain_direction": "last",
    "record_kept_token_indices": False,
}

HF_GENERATION_KWARGS = {
    "max_new_tokens": HF_MAX_NEW_TOKENS,
    "temperature": HF_TEMPERATURE,
    "top_p": HF_TOP_P,
    "do_sample": True,
    "num_return_sequences": 1,
    "return_dict_in_generate": True,
    "output_scores": False,
    "output_attentions": False,
    "output_hidden_states": False,
}

PRECOMPUTED_DIR = Path("hf_precomputed_kv")
PRECOMPUTED_DIR.mkdir(exist_ok=True)

compression_config = get_compression_config()
compression_config["method"] = DEFAULT_METHOD
compression_config["method_config"].update(METHOD_CONFIG)

if "documents" not in globals():
    documents: List[str] = [
        # """
        # Acme Corp. quarterly report indicates a 12% year-over-year growth in cloud services. The CFO attributes success to aggressive regional expansion and a revamped partner program.
        # """.strip(),
        # """
        # Technical design notes describe a retrieval-augmented generation pipeline. It highlights: (1) vector search over customer support tickets, (2) RAG responses cached for follow-up, and (3) a plan to migrate to vLLM for throughput. Key risks: stale ticket embeddings and missing observability.
        # """.strip(),
        """
        Customer interview transcript: The buyer wants faster root-cause analysis in their observability stack and prefers integrations that do not require schema changes. They have a three-month decision window.
        """.strip(),
    ]


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


def build_document_prompt(doc_id: int, document: str, context: str = "") -> str:
    
    instruction = (
        "You are processing a sequence of documents. "
        "Summarize the current document in <=5 bullet points and flag notable risks."
    )
    
    parts: List[str] = []
    if context.strip():
        parts.append(context.strip())
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


set_seed()
TOKENIZER, MODEL, DEVICE = _load_tokenizer_and_model()




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


def compute_iterative_cache(documents: List[str]) -> Dict[str, Any]:
    metadata_path = PRECOMPUTED_DIR / "metadata.json"
    if metadata_path.exists() and not os.getenv("RKV_RECOMPUTE_KV"):
        with metadata_path.open("r") as fp:
            return json.load(fp)

    tokenizer = TOKENIZER
    model = MODEL
    device = DEVICE

    summaries: List[str] = []
    context_prefix = ""
    layer_record: Dict[int, Dict[str, torch.Tensor]] = {}
    token_records: Dict[int, List[int]] = {}
    prefill_lengths: Dict[int, int] = {}

    for doc_id, document in enumerate(tqdm(documents, desc="Precomputing HF KV"), start=1):
        prompt = build_document_prompt(doc_id, document, context_prefix)
        generation_prompt = tokenizer.apply_chat_template(
			conversation =[{"role": "user", "content": prompt}],
			tokenize=False,
			add_generation_prompt=True, 
			enable_thinking=True,
    	)
        inputs = tokenizer(generation_prompt, return_tensors="pt")
        input_len = inputs["input_ids"].shape[-1]
        inputs = {k: v.to(device) for k, v in inputs.items()}
        print("input_len: ", input_len)

        with ExitStack() as stack:
            progress = TqdmProgress(total_steps=HF_MAX_NEW_TOKENS)
            stack.callback(progress.close)
            with torch.inference_mode():
                generation = model.generate(
                    **inputs,
                    **HF_GENERATION_KWARGS,
                    use_cache=True,
                    stopping_criteria=StoppingCriteriaList([progress]),
                )

        past_key_values = getattr(generation, "past_key_values", None)
        if past_key_values is None:
            raise RuntimeError(
                "Model.generate did not return past_key_values; ensure use_cache=True and return_dict_in_generate=True."
            )

        completion_slice = generation.sequences[:, input_len:].detach().cpu()
        summary = tokenizer.decode(completion_slice[0], skip_special_tokens=True).strip()
        summaries.append(summary)

        layer_record = _collect_layer_tensors(past_key_values)
        token_records[doc_id] = inputs["input_ids"][0, :input_len].detach().cpu().tolist()
        context_prefix = _append_context(context_prefix, doc_id, document, summary)

    if not layer_record:
        raise RuntimeError("No KV tensors were captured during precomputation.")

    kv_path = PRECOMPUTED_DIR / "combined_kv.pt"
    torch.save(layer_record, kv_path)

    first_entry = next(iter(layer_record.values()))
    seq_len = int(first_entry["key"].shape[2])

    metadata = {
        "kv_path": str(kv_path.resolve()),
        "seq_len": seq_len,
        "documents": documents,
        "summaries": summaries,
        "context": context_prefix,
        "context_token_ids": token_records,
    }

    with metadata_path.open("w") as fp:
        json.dump(metadata, fp, indent=2)

    print(f"Captured combined KV cache in {kv_path}.")
    return metadata


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
    for doc_id, document in enumerate(tqdm(documents, desc="Precomputing HF KV"), start=1):
        prompt = build_document_prompt(doc_id, document, context_prefix)
        generation_prompt = tokenizer.apply_chat_template(
			conversation =[{"role": "user", "content": prompt}],
			tokenize=False,
			add_generation_prompt=True, 
			enable_thinking=True,
    	)
        inputs = tokenizer(generation_prompt, return_tensors="pt")
        input_len = inputs["input_ids"].shape[-1]
        token_records[doc_id] = inputs["input_ids"][0, :input_len].detach().cpu().tolist()
        inputs = {k: v.to(device) for k, v in inputs.items()}
        past_key_values =  DynamicCache()
        with ExitStack() as stack:
            progress = TqdmProgress(total_steps=HF_MAX_NEW_TOKENS)
            stack.callback(progress.close)
            with torch.inference_mode():
                generation = model.generate(
                    **inputs,
                    **HF_GENERATION_KWARGS,
                    use_cache=True,
                    stopping_criteria=StoppingCriteriaList([progress]),
                )

                past_key_values = generation.past_key_values
                past_key_values_dict[doc_id] = past_key_values
                # torch.save(past_key_values, "past_key_values_1.pt")    
    # Get sequence length from first cache
    first_cache = next(iter(past_key_values_dict.values()))
    seq_len = first_cache.get_seq_length() if hasattr(first_cache, 'get_seq_length') else first_cache.key_cache[0].shape[2]
    
    kv_path = PRECOMPUTED_DIR / "past_key_values_dict.pt"
    torch.save(past_key_values_dict, kv_path)
    print(f"Saved {len(past_key_values_dict)} document caches to {kv_path}")
    
    metadata = {
        "kv_path": str(kv_path.resolve()),
        "seq_len": seq_len,
        "documents": documents,
        "summaries": summaries,
        "context_token_ids": token_records,
    }
    
    # Save metadata to file (will overwrite existing metadata.json)
    with metadata_path.open("w") as fp:
        json.dump(metadata, fp, indent=2)
    
    print(f"Saved metadata to {metadata_path}")
    return metadata
                
# precomputed_metadata = compute_precomputed_kv(documents)
dynamic_cache_metadata = compute_dynamic_cache(documents)