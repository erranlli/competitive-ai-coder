## Training and Model Consolidation

Scripts and configs to fine-tune Qwen-family models on Codeforces-style datasets and to consolidate/checkpoint weights for inference.

### Files

- `train_qwen_think.py`
  - TRL SFT training over chat-formatted pairs with strict formatting and label masking.
  - Supports DeepSpeed ZeRO, bf16, gradient checkpointing, evaluation with early stopping.
  - Key flags (see script for full list):
    - `--model-name`, `--dataset-path`, `--output-dir`
    - `--max-seq-length`, `--per-device-train-batch-size`, `--gradient-accumulation-steps`
    - `--deepspeed`, `--learning-rate`, `--num-train-epochs`, `--warmup-ratio`
    - `--eval-steps`, `--logging-steps`, `--validation-split-percentage`
    - `--resume-from-checkpoint`, `--disable-training-eval`
  - Early stopping is enabled only if eval is on; metric is `eval_loss` with `greater_is_better=False`.

- `deepspeed_zero2.json`, `deepspeed_zero3.json`
  - Sample ZeRO configs used by the trainer.

- `consolidate_deepspeed_checkpoint.py`
  - Converts a DeepSpeed sharded checkpoint (e.g., `checkpoint-XXXX`) into standard HF weights and saves tokenizer/config.
  - CPU-friendly, rank-0 consolidate; intended post-training.

- `consolidate_cpu.py`
  - CPU-only wrapper around DeepSpeed’s `convert_zero_checkpoint_to_fp32_state_dict`.
  - Writes a top-level `pytorch_model.bin` and saves base tokenizer/config for vLLM compatibility.

- `merge_shards_cpu.py`
  - Loads a sharded HF checkpoint on CPU and re-saves as a single weight file by setting a large `max_shard_size`.
  - Automatically detects nested shard layout like `pytorch_model.bin/` and uses it.

- `merge_lora.py`
  - Merge LoRA adapters into base weights (if applicable).

### Quickstart: Training

Example (multi-GPU with DeepSpeed):
```bash
torchrun --nproc_per_node=8 train/train_qwen_think.py \
  --model-name "Qwen/Qwen2.5-7B-Instruct" \
  --dataset-path /path/to/dataset \
  --output-dir ./runs/qwen2.5-7b \
  --deepspeed train/deepspeed_zero2.json \
  --report-to wandb --wandb-project MyProject \
  --num-train-epochs 3 --learning-rate 2e-5 --warmup-ratio 0.03
```

Tips:
- For early stopping, ensure evaluation is enabled (default) and the dataset provides a validation split.
- If you see tokenizer PAD/BOS/EOS alignment warnings, the script aligns config automatically.
- Reduce OOM during eval by lowering `--eval-max-seq-length` (if enabled) and `--per-device-eval-batch-size`.

### Consolidation for Inference

Case A: DeepSpeed ZeRO checkpoints → single FP32 file in-place:
```bash
python train/consolidate_cpu.py
```

Case B: HF sharded checkpoint → single file directory:
```bash
python train/merge_shards_cpu.py
```
Notes:
- `merge_shards_cpu.py` detects nested shard folder `pytorch_model.bin/` automatically.
- Scripts save tokenizer and config so output dirs are usable by vLLM.

### Troubleshooting

- EarlyStoppingCallback requires metric_for_best_model
  - Fixed in `train_qwen_think.py` via `metric_for_best_model="eval_loss"`, `greater_is_better=False` when eval is enabled.

- OOM during evaluation
  - Lower `--per-device-eval-batch-size`, increase `--eval-accumulation-steps` (if supported), and reduce sequence lengths.

- Shard index not found
  - Place shards under the model root or in a `pytorch_model.bin/` subfolder; `merge_shards_cpu.py` will handle either.
