# Batch Evaluation for Checkpoint Solutions

This guide explains how to evaluate solutions from multiple checkpoints in parallel using the piston checker, generating benchmark results for each checkpoint.

## Overview

The batch evaluation system processes solutions from multiple checkpoints simultaneously:
- **Parallel processing** of multiple checkpoints
- **Automatic solution file detection** (JSONL files)
- **Organized output** with separate directories for each checkpoint
- **Robust error handling** with timeout and retry mechanisms
- **Comprehensive logging** and result tracking

## Quick Start

### 1. Basic Usage (Recommended)

```bash
# Evaluate all checkpoint solutions
./eval/batch_eval_consolidated.sh
```

This will:
- Process all solutions in `/mnt/data3/model_solution_grad_norm10_16k_wu005_lr5e05_30epoch/`
- Save results to `/mnt/data3/eval_results_grad_norm10_16k_wu005_lr5e05_30epoch/`
- Use 4 parallel workers
- Generate 0 additional tests per problem

### 2. Dry Run (Check What Will Be Processed)

```bash
# See what checkpoints will be evaluated
./eval/batch_eval_consolidated.sh --dry-run
```

### 3. Custom Options

```bash
# Custom settings
./eval/batch_eval_consolidated.sh \
  --solutions-dir /path/to/your/solutions \
  --output-dir /path/to/eval/results \
  --max-generated-tests 5 \
  --max-workers 8 \
  --force
```

## Advanced Usage

### Direct Python Script

For more control, use the Python script directly:

```bash
# Evaluate all checkpoint solutions
python eval/batch_eval_solutions.py \
  --solutions-dir /mnt/data3/model_solution_grad_norm10_16k_wu005_lr5e05_30epoch \
  --output-dir /mnt/data3/eval_results_grad_norm10_16k_wu005_lr5e05_30epoch \
  --max-workers 4 \
  --max-generated-tests 0

# Evaluate specific checkpoints
python eval/batch_eval_solutions.py \
  --solutions-pattern "/mnt/data3/solutions/checkpoint-{196,392,588}" \
  --output-dir /mnt/data3/selected_eval_results \
  --max-generated-tests 5

# Custom evaluation settings
python eval/batch_eval_solutions.py \
  --solutions-dir /mnt/data3/solutions \
  --output-dir /mnt/data3/eval_results \
  --max-workers 8 \
  --max-generated-tests 10 \
  --generated-tests-workers 32 \
  --timeout 600
```

## Options Reference

### Wrapper Script (`batch_eval_consolidated.sh`)

| Option | Description | Default |
|--------|-------------|---------|
| `--solutions-dir DIR` | Directory with checkpoint solutions | `/mnt/data3/model_solution_grad_norm10_16k_wu005_lr5e05_30epoch` |
| `--output-dir DIR` | Output directory for evaluation results | `/mnt/data3/eval_results_grad_norm10_16k_wu005_lr5e05_30epoch` |
| `--max-workers N` | Max parallel evaluation workers | `4` |
| `--max-generated-tests N` | Max generated tests per problem | `0` |
| `--generated-tests-workers N` | Workers for generated tests | `16` |
| `--endpoint URL` | Evaluation endpoint | `http://localhost:2000` |
| `--timeout N` | Timeout per evaluation (seconds) | `300` |
| `--dry-run` | Show what would be processed | - |
| `--force` | Overwrite existing results | - |

### Python Script (`batch_eval_solutions.py`)

#### Input Options
- `--solutions-dir DIR` - Directory containing checkpoint solution subdirectories
- `--solutions-pattern PATTERN` - Glob pattern for solution directories
- `--solutions-list FILE` - File with list of solution directory paths
- `--checkpoint-filter PATTERN` - Filter checkpoints by name pattern

#### Processing Options
- `--max-workers N` - Maximum parallel evaluation workers (default: 4)
- `--sort-order ORDER` - Sort order: step, name, none (default: step)
- `--continue-on-error` - Continue if individual evaluations fail

#### Evaluation Parameters
- `--max-generated-tests N` - Max generated tests per problem (default: 0)
- `--generated-tests-workers N` - Workers for generated tests (default: 16)
- `--endpoint URL` - Evaluation endpoint URL (default: http://localhost:2000)
- `--timeout N` - Timeout per evaluation in seconds (default: 300)

#### Control Options
- `--dry-run` - Show what would be processed
- `--force` - Overwrite existing results
- `--continue-on-error` - Continue if individual evaluations fail

## Output Structure

Evaluation results are organized by checkpoint:

```
output_dir/
├── eval_checkpoint_49/
│   ├── eval_qwen-qwen2.5-7b-instruct__open-r1-codeforces__default__test__vllm_*.txt
│   └── eval_*.txt
├── eval_checkpoint_98/
│   ├── eval_qwen-qwen2.5-7b-instruct__open-r1-codeforces__default__test__vllm_*.txt
│   └── eval_*.txt
├── eval_checkpoint_147/
│   └── ...
├── batch_eval_results.json  # Summary of all evaluations
└── ...
```

## Evaluation Process

### For Each Checkpoint:
1. **Scan**: Find all JSONL solution files in the checkpoint directory
2. **Process**: Run evaluation on each solution file
3. **Save**: Store evaluation results in organized output files
4. **Log**: Track progress and handle errors

### For Each Solution File:
```bash
python eval/eval_with_piston_gentest_checker_stats.py \
  --solutions-path /path/to/solution.jsonl \
  --max-generated-tests 0 \
  --generated-tests-workers 16 \
  --endpoint http://localhost:2000
```

## Performance Estimates

Based on typical solution files and evaluation times:

| Configuration | Time per Checkpoint | Total Time (30 checkpoints) |
|---------------|---------------------|------------------------------|
| 4 workers, 0 generated tests | ~1 minute | ~8 minutes |
| 8 workers, 0 generated tests | ~1 minute | ~4 minutes |
| 4 workers, 5 generated tests | ~3 minutes | ~23 minutes |
| 8 workers, 5 generated tests | ~3 minutes | ~12 minutes |

## Prerequisites

### 1. Evaluation Service
Ensure the piston checker service is running:
```bash
# Check if service is running
curl http://localhost:2000

# If not running, start the service (adjust as needed)
# This depends on your piston checker setup
```

### 2. Solution Files
Ensure solution files are in the expected format:
- JSONL files with solution data
- Located in checkpoint subdirectories
- Properly formatted for the evaluation script

## Monitoring Progress

### Real-time Monitoring
```bash
# Monitor evaluation progress
watch 'find /mnt/data3/eval_results_* -name "eval_*.txt" | wc -l'

# Check batch progress
tail -f /mnt/data3/eval_results_*/batch_eval_results.json

# Monitor system resources
htop
```

### Results Analysis
```bash
# View summary
cat /mnt/data3/eval_results_grad_norm10_16k_wu005_lr5e05_30epoch/batch_eval_results.json | jq '.summary'

# Count completed evaluations
find /mnt/data3/eval_results_grad_norm10_16k_wu005_lr5e05_30epoch -name "eval_*.txt" | wc -l

# Check specific checkpoint results
ls -la /mnt/data3/eval_results_grad_norm10_16k_wu005_lr5e05_30epoch/eval_checkpoint_196/
```

## Integration Workflow

### Complete Pipeline

```bash
# 1. Consolidate checkpoints (if not done already)
python train/batch_consolidate_for_vllm.py \
  --input-dir ./ft_runs_grad_norm10_16k_wu005_lr5e05_30epoch/qwen2.5-7b \
  --output-dir /mnt/data3/all_checkpoints_grad_norm10_16k_wu005_lr5e05_30epoch \
  --all-checkpoints \
  --dtype bfloat16

# 2. Run batch inference on all checkpoints  
./infer/batch_inference_consolidated.sh

# 3. Evaluate all checkpoint solutions
./eval/batch_eval_consolidated.sh

# 4. Analyze results
python scripts/analyze_eval_results.py \
  --eval-dir /mnt/data3/eval_results_grad_norm10_16k_wu005_lr5e05_30epoch
```

## Troubleshooting

### Common Issues

1. **"No JSONL solution files found"**
   - Verify solution directory contains JSONL files
   - Check that inference completed successfully
   - Use `--dry-run` to see what's detected

2. **"Cannot connect to evaluation endpoint"**
   - Ensure piston checker service is running on port 2000
   - Check firewall settings
   - Verify endpoint URL is correct

3. **"Evaluation timeout"**
   - Increase `--timeout` value
   - Check system resources (CPU, memory)
   - Reduce `--max-workers` if system is overloaded

4. **"All evaluations failed"**
   - Check solution file format
   - Verify evaluation script is working
   - Review error logs in output directories

### Debug Mode

```bash
# Run with verbose output and single worker
python eval/batch_eval_solutions.py \
  --solutions-dir /path/to/solutions \
  --output-dir /path/to/results \
  --max-workers 1 \
  --timeout 60 \
  --dry-run
```

## Best Practices

1. **Always run dry-run first** to verify solution detection
2. **Use appropriate worker count** based on system resources
3. **Monitor evaluation endpoint** during processing
4. **Save results to fast storage** for better I/O performance
5. **Use timeout settings** appropriate for your problem complexity
6. **Check results periodically** to catch issues early

## Result Analysis

### Understanding Output Files

Each evaluation generates a text file with:
- Problem statistics
- Solution correctness
- Performance metrics
- Error details (if any)

### Comparing Across Checkpoints

```bash
# Extract accuracy scores
grep -r "Accuracy" /mnt/data3/eval_results_grad_norm10_16k_wu005_lr5e05_30epoch/

# Find best performing checkpoint
python scripts/find_best_checkpoint.py \
  --eval-dir /mnt/data3/eval_results_grad_norm10_16k_wu005_lr5e05_30epoch
```

The batch evaluation system is designed to efficiently process large numbers of checkpoint solutions while maintaining reliability and providing comprehensive result tracking.
