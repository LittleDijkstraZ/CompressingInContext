# Pipeline 文档

本目录包含 CompressingInContext 的核心流水线脚本，实现了用于上下文学习场景的 KV 缓存压缩。

## 概述

流水线包含三个主要阶段：
1. **预计算**：从训练文档生成压缩的 KV 缓存
2. **评估**：使用预计算的缓存进行下游问题求解
3. **提取**：从生成结果中提取和评估答案

## 文件结构

### 核心脚本

#### 预计算脚本

- **`precompute_cache_comp.py`**：主预计算脚本（带压缩支持）
  - 支持多种模式：`takeaways` 和 `notepad`
  - 使用 R-KV 压缩进行 KV 缓存压缩
  - 支持旋转和基于窗口的压缩
  - 使用明确的压缩调用

- **`precompute_cache_none.py`**：带压缩模式的预计算脚本（支持旋转）
  - 支持 `takeaways`、`notepad` 和 `none` 三种模式
  - `none` 模式：直接处理文档，不生成摘要
  - 支持键旋转功能（`--rotate`）
  - 可限制处理的文档数量（`--data_limit`）

- **`precompute_cache.py`**：基础预计算脚本
  - 标准预计算流程

- **`precompute_cache_notepad.py`**：Notepad 模式专用脚本
  - 只有 notepad 模式的实现

- **`precompute_cache_wo_rotation.py`**：无旋转变体
  - 不使用键旋转的压缩版本

- **`precompute_cache_comp_window_wo_rotation.py`**：基于窗口的无旋转压缩
  - 窗口压缩但不使用旋转

#### 评估脚本

- **`eval_cache.py`**：主评估脚本
  - 加载预计算的 KV 缓存
  - 使用压缩上下文生成新问题的解决方案
  - 支持流式和批量生成
  - 兼容模型：DeepSeek-R1, LEAD-7B
  - 支持多个文档 ID 的缓存加载

#### 工具脚本

- **`run_grid_search.py`**：超参数自动网格搜索
  - 测试不同的预算大小、复杂度级别和模式
  - 自动化预计算和评估流水线
  - 保存摘要和结果
  - 支持跳过验证模式

- **`extract_answer_json.py`**：从结果中提取和评估答案
  - 从 `\boxed{}` 格式或纯文本中提取答案
  - 计算 pass@k 指标
  - 生成 CSV 报告
  - 支持多种答案格式识别

- **`utils.py`**：共享工具函数
  - `PatchedDynamicCache`：扩展的 DynamicCache，支持自定义长度处理
  - `get_next_token_position_from_model`：获取下一个 token 位置
  - 缓存管理辅助函数

## 使用方法

### 1. 预计算（带压缩）

生成压缩的 KV 缓存（推荐使用）：

```bash
python -m src.pipeline.precompute_cache_comp \
    --data_path ./src/clustering/limo_clustering_results/k40/clusters/cluster_23.json \
    --budget 680 \
    --summary_complexity complex \
    --mode takeaways \
    --num_epochs 1 \
    --window_size 128 \
    --recompute
// 注意model_name是hard-coded

```

**关键参数：**
- `--budget`：KV 缓存预算大小（压缩上下文长度）
- `--summary_complexity`：提示复杂度（`simple` 或 `complex`）
- `--mode`：学习模式（`takeaways` 或 `notepad`）
- `--num_epochs`：重复文档的次数
- `--window_size`：压缩窗口大小（默认：128）
- `--recompute`：即使缓存存在也强制重新计算

### 2. 预计算（支持旋转和压缩）

使用 `precompute_cache_none.py`，支持旋转和压缩：

```bash
python -m src.pipeline.precompute_cache_none \
    --data_path ./clusteringxx.json \
    --model_name PlanePaper/LEAD-7B \
    --budget 588 \
    --mode none \
    --window_size 128 \
    --data_limit 3 \
    --rotate False \
    --target_rotation_position 3072 \
    --num_epochs 1 \
    --precomputed_dir hf_precomputed_kv_budget_588_none \
    --recompute
```

**关键参数：**
- `--data_path`：**必需**，数据文件路径
- `--model_name`：**必需**，模型名称（如 `PlanePaper/LEAD-7B`）
- `--budget`：KV 缓存预算大小（默认：680）
- `--mode`：学习模式（`takeaways`、`notepad` 或 `none`）
- `--window_size`：压缩窗口大小（默认：128）
- `--data_limit`：限制处理的文档数量（默认：3）
- `--rotate`：是否启用键旋转（默认：False）
- `--target_rotation_position`：目标旋转位置（默认：3072）
- `--num_epochs`：重复文档的次数（默认：1）
- `--precomputed_dir`：预计算缓存保存目录（可选，未指定则自动生成）
- `--recompute`：即使缓存存在也强制重新计算
- `--summary_complexity`：摘要复杂度级别（`simple` 或 `complex`，默认：complex）

**注意：**
- `--model_name` 是必需参数，必须通过命令行指定（如 `PlanePaper/LEAD-7B`）
- 使用 `--rotate True` 启用键旋转功能（需要 `target_rotation_position` 配合）
- `--data_limit` 可以限制处理的文档数量，便于快速测试
- 与 `precompute_cache_comp.py` 不同，此脚本支持通过参数指定模型名称

### 3. 评估

使用预计算的缓存进行推理：

```bash
python -m src.pipeline.eval_cache \
    --model_name deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
    --kv_cache_dir hf_precomputed_kv_budget_680_complex_takeaways \
    --max_new_tokens 8192 \
    --repeat_time 16 \
    --follow_up_prompt "你的问题 here"
```

**关键参数：**
- `--kv_cache_dir`：包含预计算缓存的目录
- `--max_new_tokens`：生成的最大 token 数
- `--repeat_time`：生成的样本数量（用于 pass@k 计算）
- `--follow_up_prompt`：要解决的问题（可选，未提供则使用默认值）
- `--doc_id`：从缓存中使用的文档 ID（默认：1）
- `--output_file`：输出结果文件路径

**输出：**
- JSON 文件包含生成结果：
  - `all_generations`：所有生成的结果列表
  - 每个结果包含：`generated_text`、`response`、`input_length`、`output_length`、`cache_seq_len`
  - `num_repeats`：重复次数
  - `streamed_chunks`：流式输出片段（如果启用）

### 4. 网格搜索

自动超参数搜索：

```bash
python -m src.pipeline.run_grid_search \
    --budget-range 352 544 800 1312 \
    --complexities complex \
    --modes takeaways notepad \
    --max-new-tokens 2048 \
    --data-path data_math.json \
    --num-epochs 1
```

**关键参数：**
- `--budget-range`：要测试的预算值列表
- `--complexities`：要测试的复杂度级别（`simple` 或 `complex`）
- `--modes`：要测试的模式（`takeaways`、`notepad` 或 `none`）
- `--max-new-tokens`：预计算时的最大新 token 数
- `--no-verify`：预计算后跳过验证
- `--cache-dir`：存储预计算缓存的目录
- `--results-dir`：存储评估结果的目录
- `--num-epochs`：文档重复次数

### 5. 答案提取

从结果文件中提取答案并计算指标：

```bash
python -m src.pipeline.extract_answer_json \
    --input_dir results/ \
    --output_csv answers.csv \
    --ground_truth data_math.json \
    --pass_k 1
```

**关键参数：**
- `--input_dir`：包含结果 JSON 文件的目录
- `--output_csv`：输出 CSV 文件路径
- `--ground_truth`：真实标签数据文件（JSON 格式）
- `--pass_k`：pass@k 的 k 值（默认：1）
- `--correct_answer`：正确答案（如果所有问题答案相同）

**答案提取优先级：**
1. `\boxed{...}` 格式的答案
2. 类似 "answer is 754" 或 "final answer: **754**" 的模式
3. 文本中的最后一个数字（回退）

## 支持的模型

- `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
- `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`
- `PlanePaper/LEAD-7B`

## 压缩方法

流水线支持 R-KV 压缩，具有以下特性：

- **预算控制**：将 KV 缓存大小限制在指定预算内
- **基于窗口的压缩**：在滑动窗口中压缩
- **旋转**：可选的键旋转以提高内存效率
- **保留策略**：可配置的 token 保留（最后、最先等）
- **明确压缩**：在生成后明确调用压缩函数

### 压缩配置

压缩参数在预计算脚本中设置：

```python
METHOD_CONFIG = {
    "budget": 680,                    # 目标缓存大小
    "window_size": 300,               # 压缩窗口大小
    "kernel_size": 7,                 # 卷积核大小
    "mix_lambda": 0.1,                # 混合参数
    "retain_ratio": 0.8,              # 保留 token 的比例
    "retain_direction": "last",       # 保留方向：last 或 first
    "rotate_keys": False,             # 是否启用键旋转
    "rotation_offset": 3072,          # 旋转偏移量
    "initial_non_compressible_length": 80,  # 初始不可压缩长度
}
```

## 模式详解

### Takeaways 模式
- **描述**：在每个文档后生成关键要点
- **特点**：
  - 跨文档累积学习
  - 每个文档独立处理并生成要点
  - 要点作为独立的摘要保存
- **适用场景**：需要明确提取每个文档的要点

### Notepad 模式
- **描述**：维护跨文档的单一演化笔记
- **特点**：
  - 维护一个全局笔记
  - 每个文档后更新笔记
  - 更紧凑的知识表示
- **适用场景**：需要维护统一的知识表示

### None 模式
- **描述**：直接处理文档，不生成摘要或笔记
- **特点**：
  - 不进行摘要生成
  - 直接处理原始文档内容
  - 可以使用压缩和旋转
- **适用场景**：需要完整文档信息但需要压缩存储的场景

## 输出格式

### 预计算输出

目录结构：
```
hf_precomputed_kv_budget_680_complex_takeaways/
├── past_key_values_dict.pt    # 压缩的 KV 缓存
└── metadata.json              # 缓存元数据
```

`metadata.json` 格式：
```json
{
  "kv_path": "path/to/past_key_values_dict.pt",
  "seq_len": 680,
  "buffer_size": 1024,
  "seen_tokens_cache": 680,
  "rotation_enabled": false,
  "rotation_offset": 0,
  "initial_non_compressible_length": 80,
  "documents": ["文档1", "文档2", ...],
  "summaries": ["摘要1", "摘要2", ...],
  "context_token_ids": [...],
  "documents": [...]
}
```

### 评估输出

JSON 格式：
```json
{
  "all_generations": [
    {
      "generated_text": "完整的生成文本（包含prompt和响应）",
      "response": "仅响应部分",
      "input_length": 100,
      "output_length": 500,
      "cache_seq_len": 680
    }
  ],
  "num_repeats": 16,
  "streamed_chunks": []
}
```

### 答案提取输出

CSV 格式：
```csv
file,answer,is_boxed,is_correct,pass_at_1,pass_at_k
result_1.json,588,True,True,1.0,...
```

## 工作流程示例

### 完整流程

**使用 precompute_cache_none.py（推荐，支持更多参数）：**

```bash
# 1. 预计算 KV 缓存（带旋转和压缩）
python -m src.pipeline.precompute_cache_none \
    --data_path data_math.json \
    --model_name PlanePaper/LEAD-7B \
    --budget 588 \
    --mode none \
    --window_size 128 \
    --data_limit 3 \
    --rotate False \
    --precomputed_dir cache_588_none \
    --recompute

# 2. 评估多个问题
python -m src.pipeline.eval_cache \
    --model_name PlanePaper/LEAD-7B \
    --kv_cache_dir cache_588_none \
    --max_new_tokens 8192 \
    --repeat_time 16 \
    --output_file results/result_588.json

# 3. 提取答案并计算指标
python -m src.pipeline.extract_answer_json \
    --input_dir results/ \
    --output_csv answers_588.csv \
    --ground_truth data_math.json
```

**使用 precompute_cache_comp.py：**

```bash
# 1. 预计算 KV 缓存（需要先修改脚本中的模型名称）
python -m src.pipeline.precompute_cache_comp \
    --data_path data_math.json \
    --budget 680 \
    --summary_complexity complex \
    --mode takeaways \
    --precomputed_dir cache_680

# 2. 评估多个问题
python -m src.pipeline.eval_cache \
    --kv_cache_dir cache_680 \
    --repeat_time 16 \
    --output_file results/result_680.json

# 3. 提取答案并计算指标
python -m src.pipeline.extract_answer_json \
    --input_dir results/ \
    --output_csv answers_680.csv \
    --ground_truth data_math.json
```

### 批量处理（网格搜索）

```bash
# 自动测试多个配置
python -m src.pipeline.run_grid_search \
    --budget-range 352 544 800 \
    --complexities complex \
    --modes takeaways notepad \
    --data-path data_math.json
```

## 使用技巧

1. **内存管理**
   - 大预算可能需要大量 GPU 内存
   - 使用 `nvidia-smi` 监控使用情况
   - 如果遇到 OOM，减小 `budget` 或 `window_size`

2. **缓存复用**
   - 预计算的缓存可以用于多次评估
   - 仅在需要时使用 `--recompute`
   - 缓存目录名称包含配置信息，便于管理

3. **网格搜索**
   - 使用网格搜索进行系统化超参数探索
   - 考虑使用多 GPU 并行执行
   - 使用 `--no-verify` 快速预计算，稍后评估

4. **答案提取**
   - 确保真实标签格式与数据格式匹配
   - 检查提取的答案是否正确识别了 `\boxed{}` 格式
   - pass@k 计算需要多个样本（`repeat_time > 1`）

5. **模式选择**
   - `takeaways`：适合需要明确要点提取的场景
   - `notepad`：适合需要统一知识表示的场景
   - `none`：适合需要完整文档信息但需要压缩的场景

6. **脚本选择**
   - `precompute_cache_none.py`：推荐使用，支持更多参数（模型名称、旋转、数据限制等），通过 `--model_name` 指定模型
   - `precompute_cache_comp.py`：模型名称硬编码在脚本中，需要修改脚本中的 `HF_MODEL_ID` 变量
   - 两者都支持压缩，但 `precompute_cache_none.py` 功能更灵活

6. **性能优化**
   - 首次运行会编译 torch graph，耗时较长
   - 后续运行会复用编译结果
   - 使用 `trust_remote_code=True` 以支持自定义模型

## 依赖项

- `transformers`：Hugging Face Transformers 库
- `torch`：PyTorch
- `rkv`：R-KV 压缩库（自定义）
- `tqdm`：进度条
- `numpy`：数值计算
- `flash-attn`：Flash Attention（可选，提升性能）

## 故障排除

### CUDA OOM（内存不足）
- **原因**：预算太大或窗口太大
- **解决**：减小 `budget` 或 `window_size`
- **检查**：`nvidia-smi` 查看 GPU 内存使用

### 缓存未找到
- **原因**：预计算未完成或路径错误
- **解决**：检查 `metadata.json` 是否存在
- **验证**：确认预计算脚本成功完成

### 模型加载问题
- **原因**：模型名称错误或缺少 `trust_remote_code`
- **解决**：检查模型名称，确保使用 `trust_remote_code=True`
- **验证**：确认模型在 Hugging Face Hub 上可用

### 压缩错误
- **原因**：R-KV 库未正确安装或配置
- **解决**：验证 R-KV 库安装，检查压缩配置参数
- **检查**：确认 `rkv.config` 和 `rkv.monkeypatch` 可导入

### 答案提取失败
- **原因**：答案格式不匹配或未找到答案
- **解决**：检查生成文本中是否包含 `\boxed{}` 或答案模式
- **调试**：查看提取的原始文本，调整正则表达式

### 生成卡住
- **原因**：模型生成过长或停止条件未触发
- **解决**：检查 `max_new_tokens` 设置，验证停止 token 配置
- **调试**：查看生成的中间输出

## 常见问题

**Q: 如何选择预算大小？**
A: 预算应该根据可用 GPU 内存和所需上下文长度选择。建议从较小值（如 512）开始，逐步增加。

**Q: Takeaways 和 Notepad 模式有什么区别？**
A: Takeaways 为每个文档生成独立的要点，Notepad 维护一个全局演化笔记。Notepad 通常更紧凑。

**Q: 为什么需要 `repeat_time > 1`？**
A: 用于计算 pass@k 指标，需要多个生成样本来评估模型的稳定性。

**Q: 如何并行运行多个实验？**
A: 使用不同的 GPU 设备（`CUDA_VISIBLE_DEVICES`）或使用网格搜索脚本。

**Q: 缓存可以跨模型使用吗？**
A: 不可以，缓存是模型特定的，每个模型需要单独的预计算。
