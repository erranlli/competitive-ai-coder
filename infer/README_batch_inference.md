# Batch Inference for Multiple Checkpoints

This guide explains how to run inference on multiple consolidated checkpoints in parallel using vLLM, maximizing GPU utilization.

## Overview

The batch inference system processes multiple checkpoints simultaneously:
- **4 GPUs per checkpoint** for optimal performance
- **2 checkpoints running concurrently** to utilize all 8 GPUs
- **Automatic GPU allocation** and management
- **Parallel processing** to minimize total inference time

## Quick Start

### 1. Basic Usage (Recommended)

```bash
# Run inference on all consolidated checkpoints
./infer/batch_inference_consolidated.sh
```

This will:
- Process all checkpoints in `/mnt/data3/all_checkpoints_grad_norm10_16k_wu005_lr5e05_30epoch/`
- Save results to `/mnt/data3/model_solution_grad_norm10_16k_wu005_lr5e05_30epoch/`
- Use 64 problems per checkpoint
- Run 2 checkpoints concurrently (4 GPUs each)

### 2. Dry Run (Check What Will Be Processed)

```bash
# See what checkpoints will be processed
./infer/batch_inference_consolidated.sh --dry-run
```

### 3. Custom Options

```bash
# Custom settings
./infer/batch_inference_consolidated.sh \
  --checkpoints-dir /path/to/your/checkpoints \
  --output-dir /path/to/results \
  --max-problems 32 \
  --force
```

## Advanced Usage

### Direct Python Script

For more control, use the Python script directly:

```bash
# Process all checkpoints
python infer/batch_inference_vllm.py \
  --checkpoints-dir /mnt/data3/all_checkpoints_grad_norm10_16k_wu005_lr5e05_30epoch \
  --output-dir /mnt/data3/model_solution_grad_norm10_16k_wu005_lr5e05_30epoch \
  --max-problems 64 \
  --gpus-per-checkpoint 4 \
  --max-concurrent 2

# Process specific checkpoints
python infer/batch_inference_vllm.py \
  --checkpoints-pattern "/mnt/data3/checkpoints/qwen2.5-7b_step_{196,392,588,784}" \
  --output-dir /mnt/data3/selected_results \
  --max-problems 64

# Custom GPU allocation
python infer/batch_inference_vllm.py \
  --checkpoints-dir /mnt/data3/checkpoints \
  --output-dir /mnt/data3/results \
  --gpus-per-checkpoint 2 \
  --max-concurrent 4 \
  --gpu-ids "0,1,2,3,4,5,6,7"
```

## Options Reference

### Wrapper Script (`batch_inference_consolidated.sh`)

| Option | Description | Default |
|--------|-------------|---------|
| `--checkpoints-dir DIR` | Directory with consolidated checkpoints | `/mnt/data3/all_checkpoints_grad_norm10_16k_wu005_lr5e05_30epoch` |
| `--output-dir DIR` | Output directory for results | `/mnt/data3/model_solution_grad_norm10_16k_wu005_lr5e05_30epoch` |
| `--max-problems N` | Max problems per checkpoint | `64` |
| `--gpus-per-checkpoint N` | GPUs per checkpoint | `4` |
| `--max-concurrent N` | Max concurrent checkpoints | `2` |
| `--dry-run` | Show what would be processed | - |
| `--force` | Overwrite existing results | - |

### Python Script (`batch_inference_vllm.py`)

#### Input Options
- `--checkpoints-dir DIR` - Directory containing checkpoint subdirectories
- `--checkpoints-pattern PATTERN` - Glob pattern for checkpoint directories
- `--checkpoints-list FILE` - File with list of checkpoint paths
- `--checkpoint-filter PATTERN` - Filter checkpoints by name pattern

#### GPU and Processing
- `--gpus-per-checkpoint N` - GPUs per checkpoint (default: 4)
- `--max-concurrent N` - Max concurrent inferences (default: 2)
- `--gpu-ids LIST` - Available GPU IDs (default: "0,1,2,3,4,5,6,7")

#### Inference Parameters
- `--max-problems N` - Problems per checkpoint (default: 64)
- `--batch-size N` - Inference batch size (default: 64)
- `--max-model-len N` - Max sequence length (default: 32768)
- `--max-new-tokens N` - Max new tokens (default: 32000)
- `--temperature F` - Sampling temperature (default: 0.1)
- `--top-p F` - Top-p sampling (default: 0.95)
- `--dtype STR` - Model dtype (default: "bfloat16")

#### Control Options
- `--dry-run` - Show what would be processed
- `--force` - Overwrite existing results
- `--continue-on-error` - Continue if individual checkpoints fail

## Output Structure

Results are organized by checkpoint:

```
output_dir/
├── checkpoint-49/
│   ├── qwen-qwen2.5-7b-instruct__open-r1-codeforces__default__test__vllm_*.jsonl
│   └── inference_log.txt
├── checkpoint-98/
│   ├── qwen-qwen2.5-7b-instruct__open-r1-codeforces__default__test__vllm_*.jsonl
│   └── inference_log.txt
├── checkpoint-147/
│   └── ...
├── batch_inference_results.json  # Summary of all runs
└── ...
```

## GPU Utilization Strategy

### Default Configuration (Optimal for 8 GPUs)
- **2 concurrent checkpoints**
- **4 GPUs per checkpoint**
- **Total: 8/8 GPUs utilized**

### Alternative Configurations

#### High Throughput (More Checkpoints)
```bash
python infer/batch_inference_vllm.py \
  --gpus-per-checkpoint 2 \
  --max-concurrent 4
# Uses: 4 concurrent checkpoints × 2 GPUs = 8 GPUs total
```

#### High Quality (Fewer Concurrent)
```bash
python infer/batch_inference_vllm.py \
  --gpus-per-checkpoint 8 \
  --max-concurrent 1
# Uses: 1 checkpoint × 8 GPUs = 8 GPUs total
```

## Performance Estimates

Based on Qwen2.5-7B with 64 problems:

| Configuration | Time per Checkpoint | Total Time (30 checkpoints) |
|---------------|---------------------|------------------------------|
| 4 GPUs/checkpoint, 2 concurrent | ~5 minutes | ~75 minutes |
| 2 GPUs/checkpoint, 4 concurrent | ~8 minutes | ~60 minutes |
| 8 GPUs/checkpoint, 1 sequential | ~3 minutes | ~90 minutes |

## Monitoring Progress

### Real-time Monitoring
```bash
# Watch GPU usage
watch nvidia-smi

# Monitor output directories
watch -n 30 'find /mnt/data3/model_solution_* -name "*.jsonl" | wc -l'

# Check batch progress
tail -f /mnt/data3/model_solution_*/batch_inference_results.json
```

### Results Summary
```bash
# View summary
cat /mnt/data3/model_solution_grad_norm10_16k_wu005_lr5e05_30epoch/batch_inference_results.json | jq '.summary'

# Count completed checkpoints
find /mnt/data3/model_solution_grad_norm10_16k_wu005_lr5e05_30epoch -name "*.jsonl" | wc -l
```

## Integration Workflow

### Complete Pipeline

```bash
# 1. Consolidate checkpoints
python train/batch_consolidate_for_vllm.py \
  --input-dir ./ft_runs_grad_norm10_16k_wu005_lr5e05_30epoch/qwen2.5-7b \
  --output-dir /mnt/data3/all_checkpoints_grad_norm10_16k_wu005_lr5e05_30epoch \
  --all-checkpoints \
  --dtype bfloat16

# 2. Run batch inference
./infer/batch_inference_consolidated.sh

# 3. Evaluate results
python eval/eval_with_piston_gentest_checker_stats.py \
  --solutions-path /mnt/data3/model_solution_grad_norm10_16k_wu005_lr5e05_30epoch/checkpoint-196/*.jsonl
```

## Troubleshooting

### Common Issues

1. **"No checkpoints found"**
   - Verify checkpoint directory exists and contains consolidated models
   - Check that directories have `model.safetensors` files
   - Use `--dry-run` to see what's detected

2. **GPU memory errors**
   - Reduce `--gpus-per-checkpoint` 
   - Reduce `--batch-size`
   - Reduce `--max-model-len`

3. **"Python environment issues"**
   - Script automatically uses `/home/g10/anaconda3/envs/rllm/bin/python`
   - Ensure this environment has required packages

4. **Slow inference**
   - Increase `--gpus-per-checkpoint` for individual speed
   - Increase `--max-concurrent` for throughput
   - Check GPU utilization with `nvidia-smi`

### Debug Mode

```bash
# Run with verbose output
python infer/batch_inference_vllm.py \
  --checkpoints-dir /path/to/checkpoints \
  --output-dir /path/to/results \
  --max-problems 1 \
  --dry-run
```

## Best Practices

1. **Always run dry-run first** to verify checkpoint detection
2. **Use --continue-on-error** for large batches to handle individual failures
3. **Monitor GPU memory** during initial runs to optimize settings
4. **Save results to fast storage** (SSD) for better I/O performance
5. **Use appropriate batch sizes** based on available GPU memory

The batch inference system is designed to maximize efficiency while maintaining reliability for processing large numbers of checkpoints.
