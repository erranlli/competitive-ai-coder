## Uploading to the Hugging Face Hub

This package contains scripts to upload models, checkpoints, datasets, and evaluation results to the Hugging Face Hub.

### Auth

Login once:
```bash
huggingface-cli login
```
Or set `HF_TOKEN` in your environment.

### Models

Upload a training run directory and (optionally) its checkpoints to a model repo under your user:
```bash
python upload_hf/upload_model_to_hub.py \
  --run-dir /mnt/data2/qwen2.5-7b-mot-full-run0 \
  --repo-name coder-ft-qwen2.5-7b \
  --upload-all-checkpoints \
  --private
```

Behavior:
- Determines your username from the token and creates `username/repo-name` if needed
- Uploads the run dir to the `main` branch
- If `--upload-all-checkpoints` is set, also uploads each `checkpoint-*` to its own branch of the same repo

If you only want to upload a single consolidated export dir:
```bash
python upload_hf/upload_model_to_hub.py \
  --run-dir /mnt/data2/last_saved_best_ckpt_single \
  --repo-name coder-ft-qwen2.5-7b
```

### Datasets

Selectively upload datasets (example maps are inside the script):
```bash
python upload_hf/upload_public_datasets.py
```
You can set `allow_patterns` in the script to upload only `data-*.arrow` files, etc.

### Evaluation results

Upload evaluation JSONL/metrics to a dataset repo:
```bash
python upload_hf/upload_eval_results_to_hub.py \
  --results-dir /root/competitive-coding-ai/results \
  --repo-id username/ccai-eval-results \
  --private
```

### Tips

- Use consolidated single-file model dirs (pytorch_model.bin or model.safetensors) when possible for simpler consumption.
- For DeepSpeed ZeRO checkpoints, consolidate first, or rely on generation-time auto-handling in `infer/generate_qwen_vllm_think.py`.
- Keep repo names short and descriptive, e.g., `coder-ft-qwen2.5-7b`.


