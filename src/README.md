# CompressingInContext Source Code

This directory contains the core implementation for the CompressingInContext project, which focuses on efficient KV cache compression for large language models.

## Overview

The basic workflow is:
1. **Precompute KV cache** for a set of documents using `pipeline/precompute_cache.py`
2. **Evaluate performance** using the cached context with `pipeline/eval_cache.py`

## Directory Structure

```
src/
├── pipeline/          # Main pipeline scripts for cache precomputation and evaluation
│   ├── precompute_cache.py          # Precompute KV cache for documents
│   ├── eval_cache.py                # Evaluate model performance with cached context
│   ├── run_grid_search.py           # Grid search over budget/length parameters
│   ├── run_grid_search_new.py       # Grid search over budget/complexity parameters
│   ├── extract_answer_json.py       # Extract and evaluate answers from results
│   └── utils.py                     # Utility classes (PatchedDynamicCache)
└── clustering/        # Document clustering utilities
    ├── limo_kmeans_clustering.py    # KMeans clustering on LIMO dataset
    ├── limo_query_cluster.py        # Query closest cluster for new questions
    └── query_example.py             # Example usage of cluster querying
```

## Pipeline Scripts

### 1. Precompute Cache (`pipeline/precompute_cache.py`)

Precomputes KV cache for a set of training documents. The model generates summaries/takeaways for each document, and the resulting KV cache is saved for later reuse.

**Usage:**
```bash
python pipeline/precompute_cache.py \
    --data_path <path_to_json_data> \
    --budget 1312 \
    --summary_complexity complex \
    --num_epochs 1 \
    --precomputed_dir <output_directory>
```

**Key Arguments:**
- `--data_path`: Path to JSON file containing documents (with `question`, `solution`, `answer` fields)
- `--budget`: KV cache budget size (default: 1312 = 1024 + 128 + 160)
- `--summary_complexity`: Complexity level for summaries (`simple` or `complex`)
- `--num_epochs`: Number of times to repeat documents
- `--precomputed_dir`: Directory to save precomputed cache (auto-generated if not specified)
- `--recompute`: Force recomputation even if cache exists

**Output:**
- `<precomputed_dir>/past_key_values_dict.pt`: Saved KV cache tensors
- `<precomputed_dir>/metadata.json`: Metadata including documents, summaries, and token IDs

**How it works (implementation notes):**
- Uses `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` patched with the R-KV compressor (`rkv.monkeypatch`) and flash-attn; compression config defaults to budget 1312 with a 128-token window and 160-token non-compressible prefix.
- Builds each training example from JSON fields (`question`, `solution`, `answer`) into a few-shot style block with `<think>` reasoning, then repeats documents `--num_epochs` times if requested.
- Prompts are wrapped in a DeepSeek chat template that asks the model to produce bullet-point takeaways (3 simple or 5 detailed when `--summary_complexity complex`) and stop on the `---` token; `StopOnTokenSequence` watches for that sentinel while `TqdmProgress` tracks generation length.
- Generation is invoked with `past_key_values` carried across iterations so the cache accumulates over all documents; the trailing stop token is trimmed from both the token sequence and the stored cache tensors to keep them aligned.
- After the loop, the script records the compressed cache (`past_key_values_dict.pt`) plus metadata (buffer size, next-token position from the R-KV tracker, summaries, token ids) to make `eval_cache.py` deterministic without re-running generation.
- The R-KV compressor supports optional key rotation (`rotate_keys`, off by default here): after compression it moves only the compressed tokens into a larger zero-padded buffer starting at `rotation_offset` (initial non-compressible tokens stay at positions `0..init_len-1`), and tracks the true next token position via `get_next_token_position()`; `pipeline/utils.get_next_token_position_from_model` reads this when saving metadata.
- Stopping reliability: `StopOnTokenSequence` now (a) matches on decoded suffixes only, with optional `delay_tokens` if you want a few tokens past the stop marker, (b) ignores stop markers that were already present in the prompt by resetting its baseline before each `generate`, and (c) retries with a slightly longer decoded tail to avoid missing a sentinel that fell just outside the minimal window. Newline-triggered compression in the R-KV patch also uses string-based suffix matching with a tiny rolling decode window for robustness.
- Debugging compression: set `RKV_DEBUG_COMPRESSION=1` (or `true`/`yes`/`on`) before running to print a line per layer whenever compression runs, e.g. `[R1KV][layer 0] compressed cache 2048 -> 1312 (rotated=False, rotation_offset=n/a, next_pos=1312)`. That is the only extra debug output aside from standard warnings/logging.

### 2. Evaluate Cache (`pipeline/eval_cache.py`)

Loads precomputed KV cache and evaluates model performance on follow-up questions.

**Usage:**
```bash
python pipeline/eval_cache.py \
    --kv_cache_dir <precomputed_directory> \
    --follow_up_prompt "Your question here" \
    --max_new_tokens 8192 \
    --repeat_time 16 \
    --output_file results/output.json
```

**Key Arguments:**
- `--kv_cache_dir`: Directory containing precomputed cache
- `--follow_up_prompt`: The question to answer using cached context
- `--max_new_tokens`: Maximum tokens to generate (default: 8192)
- `--repeat_time`: Number of generation samples (default: 1)
- `--doc_id`: Specific document ID to load (default: 1)
- `--output_file`: Path to save results

**Output:**
- JSON file with generation results, including full outputs and metadata

### 3. Grid Search Scripts

#### `run_grid_search.py`
Performs grid search over two scenarios:
- **Scenario 1**: Fixed budget sizes, varying summarization lengths
- **Scenario 2**: Fixed summarization lengths, varying budget sizes

**Usage:**
```bash
# Run both scenarios
python pipeline/run_grid_search.py --both

# Run only scenario 1
python pipeline/run_grid_search.py --scenario1 \
    --s1-budgets 192 384 768 \
    --s1-length-range 64 128 256 512 1024 2048

# Skip verification
python pipeline/run_grid_search.py --scenario1 --no-verify
```

#### `run_grid_search_new.py`
Grid search over budget sizes and prompt complexity levels with fixed summarization length.

**Usage:**
```bash
# Run with defaults
python pipeline/run_grid_search_new.py

# Custom configuration
python pipeline/run_grid_search_new.py \
    --budget-range 352 608 1312 2336 8512 \
    --complexities simple complex \
    --max-new-tokens 2048
```

### 4. Extract Answers (`pipeline/extract_answer_json.py`)

Extracts answers from batch generation results and calculates pass@k metrics.

**Usage:**
```bash
python pipeline/extract_answer_json.py
```

Edit the `results_dir` variable in the script to point to your results directory.

**Output:**
- Console output showing pass@k for each result file
- `extracted_answers_passk.csv`: Summary CSV with metrics

### 5. Utilities (`pipeline/utils.py`)

Contains `PatchedDynamicCache` class that extends HuggingFace's `DynamicCache` to allow manual sequence length override.

## Clustering Scripts

### `clustering/limo_kmeans_clustering.py`

Performs KMeans clustering on the LIMO dataset using Jina embeddings v3.

**Features:**
- Loads LIMO dataset from HuggingFace
- Generates embeddings for questions
- Performs KMeans clustering (k=16, k=40)
- Saves cluster assignments and centroids
- Generates visualization plots

**Usage:**
```bash
python clustering/limo_kmeans_clustering.py
```

### `clustering/limo_query_cluster.py`

Query system to find the closest cluster for new questions.

**Usage:**
```python
from clustering.limo_query_cluster import LIMOClusterQuery

query = LIMOClusterQuery(k_value=16)
cluster_id, samples = query.find_closest_cluster("Your question here")
```

## Typical Workflow

1. **Prepare your data** in JSON format with fields: `question`, `solution`, `answer`

2. **Precompute KV cache:**
   ```bash
   python pipeline/precompute_cache.py \
       --data_path data/train.json \
       --budget 1312 \
       --summary_complexity complex
   ```

3. **Evaluate on test questions:**
   ```bash
   python pipeline/eval_cache.py \
       --kv_cache_dir hf_precomputed_kv_budget_1312_comp_complex \
       --follow_up_prompt "Test question" \
       --repeat_time 16
   ```

4. **Extract and analyze results:**
   ```bash
   python pipeline/extract_answer_json.py
   ```

## Requirements

- PyTorch
- Transformers (HuggingFace)
- Flash Attention 2 (for efficient attention)
- Additional dependencies: numpy, tqdm, scikit-learn (for clustering)

## Notes

- The pipeline uses DeepSeek-R1-Distill-Qwen-7B by default
- KV cache compression uses the R-KV method (configured in `rkv/` module)
- Budget typically includes: base budget + window_size (128) + initial_non_compressible_length (160)
