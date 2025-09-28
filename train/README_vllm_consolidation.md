# vLLM Model Consolidation Guide

This guide explains how to consolidate DeepSpeed checkpoints and merge sharded models for vLLM inference. vLLM requires models to be in single-file format without sharded tensors.

## Overview

We provide two main scripts:

1. **`consolidate_for_vllm.py`** - Unified script for single model/checkpoint consolidation
2. **`batch_consolidate_for_vllm.py`** - Batch processing for multiple models/checkpoints

## Quick Start

### Single Model Consolidation

```bash
# Consolidate latest DeepSpeed checkpoint
python train/consolidate_for_vllm.py \
  --input ./ft_runs/qwen2.5-7b \
  --output ./models/qwen2.5-7b-consolidated

# Consolidate specific checkpoint
python train/consolidate_for_vllm.py \
  --input ./ft_runs/qwen2.5-7b/checkpoint-196 \
  --output ./models/qwen2.5-7b-step196

# Merge HuggingFace sharded model
python train/consolidate_for_vllm.py \
  --input ./sharded_model \
  --output ./single_file_model \
  --type hf_sharded
```

### Batch Processing

```bash
# Process all training runs
python train/batch_consolidate_for_vllm.py \
  --input-pattern "./ft_runs_*/qwen2.5-7b" \
  --output-dir ./consolidated_models

# Process all checkpoints from a run
python train/batch_consolidate_for_vllm.py \
  --input-dir ./ft_runs_grad_norm10_16k_wu005_lr5e05_30epoch/qwen2.5-7b \
  --output-dir ./all_checkpoints \
  --all-checkpoints
```

## Detailed Usage

### consolidate_for_vllm.py

#### Basic Options

- `--input DIR` - Input directory (required)
- `--output DIR` - Output directory (required)
- `--type {deepspeed,hf_sharded,auto}` - Input type (default: auto)
- `--force` - Overwrite existing output

#### DeepSpeed Options

- `--checkpoint-dir DIR` - Specific checkpoint (overrides auto-detection)
- `--base-model-name NAME` - Base model for tokenizer/config (default: Qwen/Qwen2.5-7B-Instruct)

#### Model Options

- `--dtype {float32,float16,bfloat16}` - Output precision (default: float32)
- `--max-shard-size SIZE` - Max shard size (default: 40GB)
- `--keep-original-precision` - Don't convert precision
- `--safetensors` - Save as safetensors format (default: True)
- `--pytorch-bin` - Save as PyTorch .bin format instead

#### Validation

- `--validate-vllm` - Validate vLLM compatibility (default: true)

### batch_consolidate_for_vllm.py

#### Input Selection

Choose one:
- `--input-dir DIR` - Single directory
- `--input-pattern PATTERN` - Glob pattern for multiple directories
- `--input-list FILE` - File with directory list

#### Processing Options

- `--all-checkpoints` - Process all checkpoints (not just latest)
- `--checkpoint-filter PATTERN` - Filter checkpoints by pattern
- `--max-workers N` - Parallel workers (default: 4)
- `--naming-template TEMPLATE` - Output naming template (default: `{input_name}_step_{step}`)

#### Control Options

- `--dry-run` - Show what would be processed
- `--continue-on-error` - Don't stop on individual failures
- `--force` - Overwrite existing outputs

## Input Types and Auto-Detection

The script automatically detects:

### DeepSpeed Checkpoints
- Directories with `zero_to_fp32.py`
- Directories with `mp_rank_XX_model_states.pt` files
- Training run directories with `checkpoint-*` subdirectories

### HuggingFace Sharded Models
- Directories with `pytorch_model.bin.index.json`
- Nested shards in `pytorch_model.bin/` subdirectory

### Single-File Models
- Directories with `pytorch_model.bin` or `model.safetensors`

## Output Structure

Consolidated models will have:
```
output_dir/
├── model.safetensors         # Single consolidated weights file (safetensors format)
├── config.json              # Model configuration
├── tokenizer.json           # Tokenizer
├── tokenizer_config.json    # Tokenizer config
├── special_tokens_map.json  # Special tokens
└── ...                      # Other tokenizer files
```

**Note**: By default, models are saved in the modern `safetensors` format, which is:
- ✅ Faster to load
- ✅ More secure (no arbitrary code execution)
- ✅ Better memory efficiency
- ✅ Preferred by vLLM and modern ML frameworks

Use `--pytorch-bin` if you specifically need the older `.bin` format.

## Examples

### Example 1: Basic DeepSpeed Consolidation

```bash
# Your training output
./ft_runs/qwen2.5-7b/
├── checkpoint-100/
├── checkpoint-200/
└── checkpoint-300/

# Consolidate latest checkpoint
python train/consolidate_for_vllm.py \
  --input ./ft_runs/qwen2.5-7b \
  --output ./models/qwen2.5-7b-final

# Result ready for vLLM
./models/qwen2.5-7b-final/
├── model.safetensors     # Single 13GB file (safetensors format)
├── config.json
└── tokenizer files...
```

### Example 2: Batch Processing Multiple Runs

```bash
# Multiple training experiments
./experiments/
├── ft_runs_grad_norm05_16k/qwen2.5-7b/
├── ft_runs_grad_norm10_16k/qwen2.5-7b/
└── ft_runs_grad_norm10_32k/qwen2.5-7b/

# Process all experiments
python train/batch_consolidate_for_vllm.py \
  --input-pattern "./experiments/ft_runs_*/qwen2.5-7b" \
  --output-dir ./consolidated_experiments \
  --naming-template "{run_name}_final"

# Results
./consolidated_experiments/
├── ft_runs_grad_norm05_16k_final/
├── ft_runs_grad_norm10_16k_final/
└── ft_runs_grad_norm10_32k_final/
```

### Example 3: Processing Specific Checkpoints

```bash
# Process multiple specific checkpoints
python train/batch_consolidate_for_vllm.py \
  --input-dir ./ft_runs_grad_norm10_16k_wu005_lr5e05_30epoch/qwen2.5-7b \
  --output-dir ./checkpoint_comparison \
  --checkpoint-filter "checkpoint-{100,200,300}" \
  --naming-template "qwen25_step_{step}"

# Results (note: default naming template now includes step numbers)
./checkpoint_comparison/
├── qwen25_step_100/
├── qwen25_step_200/
└── qwen25_step_300/
```

## Using with vLLM

After consolidation, use the model with vLLM:

```python
from vllm import LLM, SamplingParams

# Load consolidated model
llm = LLM(
    model="./models/qwen2.5-7b-consolidated",
    tensor_parallel_size=8,
    dtype="bfloat16"
)

# Generate
sampling_params = SamplingParams(temperature=0.1, top_p=0.95, max_tokens=1000)
outputs = llm.generate(prompts, sampling_params)
```

Or with the vLLM server:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model ./models/qwen2.5-7b-consolidated \
  --tensor-parallel-size 8 \
  --dtype bfloat16
```

## Troubleshooting

### Common Issues

1. **"Model is still sharded" error**
   - The consolidation didn't create a single file
   - Try increasing `--max-shard-size` to `100GB`
   - Check if input was properly detected

2. **Out of memory during consolidation**
   - Use `--dtype float16` or `--dtype bfloat16`
   - Process on a machine with more RAM
   - Use `--keep-original-precision` if model was trained in lower precision

3. **Missing tokenizer files**
   - For DeepSpeed: ensure `--base-model-name` matches your training base model
   - For HF models: ensure tokenizer files exist in input directory

4. **vLLM compatibility issues**
   - Run with `--validate-vllm` to check compatibility
   - Ensure no `.index.json` files in output
   - Check that weights are in single `pytorch_model.bin` file

### Validation

The script validates vLLM compatibility by checking:
- ✅ Single weight file (no sharding)
- ✅ Required config files present
- ✅ No index files indicating sharding
- ⚠️ Tokenizer files present (warning if missing)

### Performance Tips

1. **Use appropriate precision**: `float16` or `bfloat16` for inference, `float32` for maximum accuracy
2. **Parallel processing**: Use `--max-workers` for batch processing
3. **Storage**: Ensure sufficient disk space (models can be 13GB+ each)
4. **Memory**: Consolidation requires loading full model in RAM

## Integration with Your Workflow

Based on your training setup, here are recommended workflows:

### After Training Completion

```bash
# Consolidate the final checkpoint
python train/consolidate_for_vllm.py \
  --input ./ft_runs_grad_norm10_32k_wu10_lr5e05_30epoch/qwen2.5-7b \
  --output ./models/qwen2.5-7b-final \
  --dtype bfloat16

# Use in your inference script
python infer/generate_qwen_vllm_think.py \
  --model-name ./models/qwen2.5-7b-final \
  --batch-size 64 \
  --tensor-parallel-size 8 \
  # ... other args
```

### Checkpoint Comparison

```bash
# Consolidate multiple checkpoints for comparison
python train/batch_consolidate_for_vllm.py \
  --input-dir ./ft_runs_grad_norm10_16k_wu005_lr5e05_30epoch/qwen2.5-7b \
  --output-dir ./checkpoint_comparison \
  --checkpoint-filter "checkpoint-{50,100,150,196}" \
  --naming-template "qwen25_step_{step}" \
  --dtype bfloat16

# Test each checkpoint
for checkpoint in ./checkpoint_comparison/*/; do
    echo "Testing $checkpoint"
    python infer/generate_qwen_vllm_think.py \
      --model-name "$checkpoint" \
      --max-problems 10 \
      # ... other args
done
```

## File Sizes and Storage

Typical file sizes for Qwen2.5-7B:
- **Original sharded**: ~13GB across multiple files
- **Consolidated float32 (safetensors)**: ~26GB single file
- **Consolidated bfloat16 (safetensors)**: ~13GB single file
- **Consolidated float16 (safetensors)**: ~13GB single file

**Safetensors vs PyTorch .bin**:
- Safetensors files are typically 5-10% smaller due to better compression
- Loading is 2-3x faster with safetensors
- vLLM has optimized support for safetensors format

Plan storage accordingly, especially for batch processing multiple checkpoints.
