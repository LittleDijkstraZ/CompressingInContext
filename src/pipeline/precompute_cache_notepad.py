import argparse
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "8"
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


class StopOnTokenSequence(StoppingCriteria):
    """
    Stops generation when any provided stop sequence appears as a decoded suffix,
    optionally after a delay of `delay_tokens` tokens.

    Accepts raw strings or token id sequences (single list[int] or list of list[int]).
    """

    def __init__(self, stop_sequences: List[List[int]] | List[int] | List[str], tokenizer: AutoTokenizer, delay_tokens: int = 0):
        if stop_sequences and isinstance(stop_sequences[0], int):
            stop_sequences = [stop_sequences]  # type: ignore[list-item]

        self.tokenizer = tokenizer
        self._base_len: int | None = None
        self.stop_strings: List[str] = []
        self.stop_strings_stripped: List[str] = []
        self.max_window_tokens = 0
        self.delay_tokens = max(0, delay_tokens)
        self._first_match_len: int | None = None

        for seq in stop_sequences:
            if not seq:
                continue
            if isinstance(seq, str):
                stop_text = seq
                token_ids = tokenizer.encode(stop_text, add_special_tokens=False)
            else:
                token_ids = list(seq)
                stop_text = tokenizer.decode(
                    token_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            if not token_ids:
                continue
            self.stop_strings.append(stop_text)
            stripped = stop_text.rstrip()
            if stripped:
                self.stop_strings_stripped.append(stripped)
            self.max_window_tokens = max(self.max_window_tokens, len(token_ids))

        self.stop_suffixes = tuple(self.stop_strings)
        self.stop_suffixes_stripped = tuple(self.stop_strings_stripped)

    def reset(self, start_len: int | None = None):
        """Reset internal counters for a new generation call."""
        self._first_match_len = None
        self._base_len = start_len if start_len is not None else 0

    def _matched(self, text: str) -> bool:
        if text.endswith(self.stop_suffixes):
            return True
        stripped = text.rstrip()
        if stripped and stripped.endswith(self.stop_suffixes_stripped):
            return True
        return False

    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> bool:
        if not self.stop_suffixes or self.max_window_tokens == 0:
            return False

        cur_len = input_ids.shape[-1]
        if self._base_len is None:
            # Fallback: initialize on first call if reset wasn't invoked.
            self._base_len = cur_len
        if cur_len <= (self._base_len or 0):
            return False

        if cur_len == 0:
            return False

        window = min(cur_len, self.max_window_tokens)
        last_tokens = input_ids[:, -window:]

        decoded_suffixes = self.tokenizer.batch_decode(
            last_tokens.tolist(),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        saw_match = any(self._matched(text) for text in decoded_suffixes)

        if saw_match and self._first_match_len is None:
            self._first_match_len = cur_len
            print(f"[StopOnTokenSequence] matched stop suffix at len={cur_len} (delay={self.delay_tokens})")

        if self._first_match_len is None:
            return False

        if self.delay_tokens == 0:
            print(f"[StopOnTokenSequence] stopping immediately at len={cur_len}")
            return True

        elapsed = cur_len - self._first_match_len
        remaining = self.delay_tokens - elapsed

        if remaining > 0:
            print(f"[StopOnTokenSequence] stop matched; waiting {remaining} more token(s) (len={cur_len})")
            return False

        print(f"[StopOnTokenSequence] delay satisfied; stopping at len={cur_len}")
        return True



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

    # newline_candidates = ["\n", ".\n", ")\n", "\n\n", ".\n\n", ")\n\n"]
    # newline_candidates = ["\n", ".\n", ")\n", "\n\n", ".\n\n", ")\n\n"]
    # newline_candidates = ['---', '\n\n---\n\n', '\n\n---\n', '.\n\n---\n\n', '\n---\n', ]
    newline_candidates = ['---', '---\n', '---\n\n', '<｜end▁of▁sentence｜>']

    newline_token_ids: List[int] = []
    newline_strings: List[str] = []
    newline_max_len = 1
    for pattern in newline_candidates:
        ids = tokenizer.encode(pattern, add_special_tokens=False)
        if ids:
            newline_token_ids.append(ids[-1])
            newline_strings.append(pattern)
            newline_max_len = max(newline_max_len, len(ids))
    model.newline_token_ids = newline_token_ids
    model._newline_tokenizer = tokenizer
    model._newline_strings = newline_strings
    model._newline_max_len = newline_max_len

    think_ids = tokenizer.encode("</think>", add_special_tokens=False)
    model.after_think_token_ids = [think_ids[-1]] if think_ids else []

    HF_GENERATION_KWARGS.setdefault("pad_token_id", tokenizer.pad_token_id)
    HF_GENERATION_KWARGS.setdefault("eos_token_id", tokenizer.eos_token_id)

    return tokenizer, model, device


def apply_chat_template(input_text, model_name: str, append_instruction=False, is_first_document=True) -> str:
    if model_name == 'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B':
        bos = "<｜begin▁of▁sentence｜>"
        user_token = "<｜User｜>"
        assistant_token = "<｜Assistant｜>"
        # think_end_think = "<think>\n</think>"
        think_end_think = "<think>\nOkey, my learning begins:"

    else:
        raise ValueError(f"Unsupported model for chat template: {model_name}")

    prompt = (
        "You are currently learning from some examples, which will later help you to solve similar problems. "
        "Given the problem, reasoning, and solution, you will maintain a note that captures key insights and strategies. "
        "After each new problem, you should update your note (under the ###Note section) to incorporate new learnings while keeping it concise. "
    )

    # Different backbone for first document vs subsequent documents
    if is_first_document:
        cur_backbone = (
            "(Given the problem, reasoning, and solution, I will create an initial note capturing the key insights "
            "and strategies that can help me solve similar problems in the future. "
        )
    else:
        cur_backbone = (
            "(Given the new problem, reasoning, and solution, I will now modify and update my existing note "
            "to incorporate new insights from this problem while keeping the note concise and comprehensive. "
        )

    Instruction_simple = (
        "The note should be bullet points. They should be highlevel, short and to the point. "
        "We should have 3 bullet points (i.e. 1., 2., 3.), following a markdown format. I will end the note with the '---' token underneath the last point.)\n1."
    )
    # Instruction_medium = (
    #     "These takeaways should be bullet points. "
    #     "You should have 5 bullet points (i.e. 1., 2., 3., 4., 5.). "
    #     "For each bullet point, you should have 2-3 sub-bullet points. (i.e. 1.1., 1.2., 1.3)"
    # )
    Instruction_complex = (
        "The note should be bullet points. "
        "I should first identify and describe the category of these problems that I've just solved. "
        "Then, I need to to write down thekey insights and strategies "
        "that can help me solve similar problems in the future. "
        "If there is a previous note, I need to update it with the new insights and strategies, "
        "keeping points from previous note or adding points to it that are commonly applicable to similar problems."
        "I must include specific steps and details from the reasoning process. "
        "I will generate 5 points (i.e. 1., 2., 3., 4., 5.) and end the note with '---' token underneath the last point.)\n1."
    )
    if SUMMARY_COMPLEXTIY == "simple":
        # prompt += Instruction_simple
        cur_backbone += Instruction_simple
    # elif SUMMARY_COMPLEXTIY == "medium":
    #     prompt += Instruction_medium
    elif SUMMARY_COMPLEXTIY == "complex":
        # prompt += Instruction_complex
        cur_backbone += Instruction_complex

    if append_instruction:
        input_text += cur_backbone
    generation_prompt = f"""{bos}{user_token}{prompt}
{assistant_token}{think_end_think}{input_text}"""
    return generation_prompt, input_text


class DynamicCacheWithCustomizedLength(DynamicCache):
    def __init__(self, ):
        super().__init__()
        self._rotation_enabled = False  # Flag to indicate if rotation has been applied

    def get_seq_length(self, layer_idx: Optional[int] = 0):
        return self._seen_tokens

    def set_customized_length(self, customized_length: int):
        """
        Manually set _seen_tokens to a specific value.
        This is used after rotation to set the correct position.
        """
        print(f"[CACHE] set_customized_length: {customized_length}, enabling rotation mode")
        self._seen_tokens = customized_length
        # Enable rotation mode - from now on, _seen_tokens will be manually controlled
        self._rotation_enabled = True

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """
        Override update to prevent automatic _seen_tokens increment when rotation is enabled.
        """
        # Store the current _seen_tokens value
        old_seen_tokens = self._seen_tokens if hasattr(self, '_seen_tokens') else 0

        # Call parent update to add keys/values to cache
        result = super().update(key_states, value_states, layer_idx, cache_kwargs)

        # If rotation is enabled, manually control _seen_tokens
        if self._rotation_enabled and layer_idx == 0:
            # The parent update() automatically incremented _seen_tokens
            # We need to manually control it instead
            new_tokens = key_states.shape[-2]
            expected_seen_tokens = old_seen_tokens + new_tokens
            if self._seen_tokens != expected_seen_tokens:
                print(f"[CACHE UPDATE] Layer 0: Correcting _seen_tokens: {self._seen_tokens} -> {expected_seen_tokens} (added {new_tokens} tokens, rotation_enabled={self._rotation_enabled})")
            self._seen_tokens = expected_seen_tokens

        return result



def compute_dynamic_cache(documents: List[str], recompute: bool = False) -> Dict[str, Any]:
    metadata_path = PRECOMPUTED_DIR / "metadata.json"
    if metadata_path.exists() and not recompute and not os.getenv("RKV_RECOMPUTE_KV"):
        with metadata_path.open("r") as fp:
            return json.load(fp)

    tokenizer = TOKENIZER
    model = MODEL
    device = DEVICE
    summaries: List[str] = []
    context_prefix = ""
    past_key_values_dict: Dict[int, DynamicCacheWithCustomizedLength] = {}
    token_records: Dict[int, List[int]] = {}
    past_conversation = []
    past_key_values =  DynamicCacheWithCustomizedLength()
    past_context = ''

    # stopping_tokens = ["</think>", "###", "##", '---', '\n\n6', '.\n\n6']
    # stopping_tokens = ["</think>", "###", "##", '\n\n6', '.\n\n6']

    # stopping_tokens = ['---', '\n\n---\n\n', '\n\n---\n', '.\n\n---\n\n', '\n---\n', '\n---\n\n']
    # stopping_tokens =  ['\n\n---\n\n', '\n\n---\n', '.\n\n---\n\n', '\n---\n', ]
    # stopping_tokens =  ['---', '\n\n---\n\n', '\n\n---\n', '.\n\n---\n\n', '\n---\n', ]
    stopping_tokens =  ['---', '\n6.']


    stop_sequences = [
        tokenizer.encode(token, add_special_tokens=False) for token in stopping_tokens
    ]

    stop_on_any_sequence = StopOnTokenSequence(stop_sequences, tokenizer, delay_tokens=2) if stop_sequences else None

    # Track the current note that gets updated each iteration
    current_note = ""

    # Track the previous input text to compute only new tokens
    previous_input_text = ""
    # Track the saved _seen_tokens value from the previous round (after set_customized_length)
    saved_seen_tokens = None

    for doc_id, example in enumerate(tqdm(documents, desc="Precomputing HF KV")):
        is_first_document = (doc_id == 0)

        # Build the context: document + previous note (if exists) + prompt to update
        past_context += example + "\n"

        if not is_first_document and current_note:
            # Append the previous note and ask the model to modify it
            past_context += f"\n###Previous Note:\n{current_note}\n"

        past_context += "\n###Note:\n"

        input_text, past_context = apply_chat_template(
            input_text = past_context,
            model_name = HF_MODEL_ID,
            append_instruction=True,
            is_first_document=is_first_document,
        )
        # print("=="*100)
        # print("History: \n", input_text)
        # print("=="*100)

        print("input_text: ", input_text)

        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[-1]
        # Update previous_input_text for next iteration
        previous_input_text = input_text

        with ExitStack() as stack:
            progress = TqdmProgress(total_steps=HF_MAX_NEW_TOKENS)
            stack.callback(progress.close)
            stopping_criteria = [progress]
            if stop_on_any_sequence:
                stop_on_any_sequence.reset(start_len=inputs["input_ids"].shape[-1])
                stopping_criteria.append(stop_on_any_sequence)

            print(f"prefill size: {inputs['input_ids'].shape[-1] - past_key_values._seen_tokens}")
            with torch.inference_mode():
                generation = model.generate(
                    **inputs,
                    **HF_GENERATION_KWARGS,
                    use_cache=True,
                    return_dict_in_generate=True,
                    stopping_criteria=StoppingCriteriaList(stopping_criteria),
                    past_key_values=past_key_values,
                )

                past_key_values = generation.past_key_values
                print("=="*100)
                print(f"cur cache size: {past_key_values.key_cache[0].shape[2]}")
                num_seen_tokens = len(generation.sequences[0, :].detach().cpu())

                effective_seen_tokens = num_seen_tokens
                print(f"seen tokens: {num_seen_tokens}")
                past_key_values.set_customized_length(effective_seen_tokens)

        # assume only one sequence
        full_text_ids = generation.sequences[0, :].detach().cpu()

        # truncate last token if it's eos_token. else, just leave it as a broken sentence.
        # last_token = full_text_ids[-1]
        # if last_token == tokenizer.eos_token_id or tokenizer.decode(last_token) in stopping_tokens:
        #     full_text_ids = full_text_ids[:-1]
        #     for layer_idx in range(len(past_key_values.key_cache)):
        #         past_key_values.key_cache[layer_idx] = past_key_values.key_cache[layer_idx][:, :, :-1, :]
        #         past_key_values.value_cache[layer_idx] = past_key_values.value_cache[layer_idx][:, :, :-1, :]

        # always remove last
        full_text_ids = full_text_ids[:-1]
        for layer_idx in range(len(past_key_values.key_cache)):
            past_key_values.key_cache[layer_idx] = past_key_values.key_cache[layer_idx][:, :, :-1, :]
            past_key_values.value_cache[layer_idx] = past_key_values.value_cache[layer_idx][:, :, :-1, :]


        # Extract the generated note
        generated_note = tokenizer.decode(full_text_ids[input_len:], skip_special_tokens=True).strip()
        summaries.append(generated_note)
        raw = tokenizer.decode(full_text_ids[input_len:], skip_special_tokens=False)

        # Update the current note for the next iteration
        current_note = generated_note

        print(f"raw length={len(raw)} vs note length={len(generated_note)}")
        print(f"Warning: raw!=note") if raw != generated_note else None
        print("=" * 100)
        print(f"Document {doc_id} - Generated/Updated Note:")
        print(tokenizer.decode(full_text_ids[input_len:], skip_special_tokens=False))
        print("=" * 100)

        print(f"cache_length= {past_key_values.key_cache[0].shape[2]} vs full_text_ids_tokenized_len= {len(full_text_ids)}")

        past_context += raw + "\n" # add a line switch at the end. the model will also encode this


    # Get sequence length from first cache
    cache_len = past_key_values.key_cache[0].shape[2]
    print(f"cached seq_len (buffer size): {cache_len}")
    print(f"total_input_tokens: {input_len}")

    past_key_values_dict[1] = past_key_values

    kv_path = PRECOMPUTED_DIR / "past_key_values_dict.pt"
    torch.save(past_key_values_dict, kv_path)
    print(f"Saved {len(past_key_values_dict)} document caches to {kv_path}")


    input_text, _ = apply_chat_template(
        input_text = past_context,
        model_name = HF_MODEL_ID,
    )
    print("text to cache: ")
    print(input_text)
    # context_token_ids = tokenizer.encode(input_text)
    context_token_ids = full_text_ids.tolist()
    print(f"Total context tokens to cache: {len(context_token_ids)}")
    print(f"cache length=", cache_len)

    metadata = {
        "kv_path": str(kv_path.resolve()),
        "seq_len": effective_seen_tokens,  # Use _seen_tokens from cache (accounts for rotation)
        "buffer_size": cache_len,  # The actual cache buffer size
        "initial_non_compressible_length": METHOD_CONFIG.get("initial_non_compressible_length", 0),
        "documents": documents,
        "notes": summaries,  # List of notes, each updated from the previous
        "final_note": current_note,  # The final consolidated note
        "context_token_ids": context_token_ids,
    }
    
    # Save metadata to file (will overwrite existing metadata.json)
    with metadata_path.open("w") as fp:
        json.dump(metadata, fp, indent=2)
    
    print(f"Saved metadata to {metadata_path}")
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute KV cache for documents")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the data file")
    parser.add_argument("--budget", type=int, default=700+80, help="KV cache budget size")
    parser.add_argument("--summary_complexity", type=str, default="complex",
                        choices=["simple", "complex"], help="Summary complexity level")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of times to repeat documents")
    parser.add_argument("--precomputed_dir", type=str, default=None,
                        help="Directory to save precomputed cache (auto-generated if not specified)")
    parser.add_argument("--recompute", action="store_true",
                        help="Force recomputation even if cache exists")
    
    return parser.parse_args()


if __name__ == "__main__":

    # os.environ["CUDA_VISIBLE_DEVICES"] = "9"

    args = parse_args()

    HF_MODEL_ID = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    ATTN_IMPL = "flash_attention_2"
    # ATTN_IMPL = "sdpa"
    # HF_MAX_NEW_TOKENS = int(os.getenv("HF_MAX_NEW_TOKENS", "64"))
    HF_MAX_NEW_TOKENS = 2048
    SUMMARY_COMPLEXTIY = args.summary_complexity
    HF_TEMPERATURE = 0.3
    HF_TOP_P = 0.95
    HF_MAX_COMPLETIONS_PER_CALL = 1

    DEFAULT_METHOD = "rkv"
    METHOD_CONFIG = {
        "budget": args.budget,
        "window_size": 200, # BUG: IDK why budget need to be > window_size here
        "kernel_size": 7,
        "mix_lambda": 0.1,
        "retain_ratio": 0.8,
        "retain_direction": "last",
        "record_kept_token_indices": False,
        "initial_non_compressible_length": 80, # skip instructions
        # Memory optimization parameters to avoid OOM
        "similarity_chunk_size": 2048,  # Reduce from 1024 to be more conservative
        "use_random_projection": False,  # Set to True if still OOM
        "projection_dim": 128,  # Only used if use_random_projection=True
        "rotate_keys": False,
        "target_rotation_position": 512,
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

    budget = METHOD_CONFIG['budget']
    max_len = HF_MAX_NEW_TOKENS
    num_epochs = args.num_epochs

    if args.precomputed_dir:
        PRECOMPUTED_DIR = args.precomputed_dir
    elif num_epochs == 1:
        PRECOMPUTED_DIR = f"hf_precomputed_kv_budget_{budget}_comp_{SUMMARY_COMPLEXTIY}"
    else:
        PRECOMPUTED_DIR = f"hf_precomputed_kv_budget_{budget}_comp_{SUMMARY_COMPLEXTIY}_epochs_{num_epochs}"

    # Check if cache already exists
    metadata_path = Path(os.path.join(PRECOMPUTED_DIR, "metadata.json"))
    if Path(PRECOMPUTED_DIR).exists() and metadata_path.exists() and not args.recompute:
        print(f"Cache directory {PRECOMPUTED_DIR} already exists with metadata.json. Skipping precomputation.")
        with metadata_path.open("r") as fp:
            dynamic_cache_metadata = json.load(fp)
    else:
        os.makedirs(PRECOMPUTED_DIR, exist_ok=True)
        PRECOMPUTED_DIR = Path(PRECOMPUTED_DIR)

        compression_config = get_compression_config()
        compression_config["method"] = DEFAULT_METHOD
        # compression_config["method_config"].update(METHOD_CONFIG)
        # compression_config['divide_method'] = 'newline'

        compression_config = {
            "method": DEFAULT_METHOD,
            "method_config": METHOD_CONFIG,
            "compression": False,
            "update_kv": True,
            "compression_content": "all",
            "divide_method": "newline",
            "divide_length": 256,
        }


        set_seed()
        TOKENIZER, MODEL, DEVICE = _load_tokenizer_and_model()

        import json
        with open(args.data_path, 'r') as f:
            data = json.load(f)

        documents = []
        for idx, item in enumerate(data[:3]):
            # sample = (
            #     f"###Problem:\n---\n{item['question']}\n---\n"
            #     f"###Reasoning:\n---\n<think>\n{item['solution']}\n</think>\nThe answer is" + '\\boxed' + f"{item['answer']}" "}\n---\n"
            # )
            sample = (
                f"###Problem:\n---\n{item['problem']}\n---\n"
                f"###Reasoning:\n---\n<think>\n{item['reasoning']}\n</think>\n---\n"
                f"###Solution:\n---\n{item['solution']}\n---\n"
            )
            sample_len = len(TOKENIZER(sample).input_ids)
            if sample_len > 12000:
                print(f"Sample {idx} is too long: {sample_len} tokens. Skipping.")
                continue

            documents.append(sample.strip())

        # Repeat documents num_epochs times
        if num_epochs > 1:
            documents = documents * num_epochs

        dynamic_cache_metadata = compute_dynamic_cache(documents, recompute=args.recompute)
