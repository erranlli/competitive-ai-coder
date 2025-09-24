## model_util: Shared Utilities for Training and Inference

This package centralizes helpers used by `train/` and `infer/` so code stays small, readable, and consistent.

### Modules

- `model_util/file_util.py`
  - `ensure_dir(path)`: mkdir -p
  - `resolve_outfile(model_name, dataset_name, subset, split)`: standardized JSONL name for generation outputs
  - `latest_checkpoint(run_dir)`: newest `checkpoint-*` under a training run
  - `is_valid_trainer_checkpoint_dir(path)`: checks for `trainer_state.json`
  - `latest_valid_trainer_checkpoint(run_dir)`: newest checkpoint containing `trainer_state.json`
  - `normalize_single_file(src_dir, dst_dir)`: copy single weight file + config/tokenizer
  - `prepare_model_dir(checkpoint_path, model_name, ds_convert)`: robust resolver for vLLM model path; handles DeepSpeed ZeRO export, nested shards, and HF single-file dirs

- `model_util/model_util.py`
  - `get_model_config_fields(model_name) -> (num_heads, num_kv_heads, max_position_embeddings)`
  - `detect_model_type(model_name) -> str`: e.g., `qwen2.5`, `qwen3`, `deepseek-r1`, `generic`
  - `load_tokenizer(model_name)`: sets pad_token and left padding for decoder-only models

- `model_util/text_util.py`
  - `normalize_text(text)`: trim trailing spaces per line
  - `extract_code_from_text(generated, language_hint='python')`: extract first fenced block or returns the input

- `model_util/train_util.py`
  - `build_messages_from_row(row)`: normalize `messages` or `input`/`output` into a 2-turn conversation
  - `enforce_strict_and_length(ds, tokenizer, max_seq_length, split_name)`: filters dataset rows to required structure and length
  - `get_map_and_tokenize_row(tokenizer, max_seq_length, enable_loss_reweight, loss_weight_code, loss_weight_noncode, mask_noncode) -> fn(row)`: single-pass tokenization with optional code-token weighting
  - `SingleLineMetricsCallback(trainer_ref)`: prints merged metrics (loss, grad_norm, learning_rate, mean_token_accuracy, entropy, num_tokens, epoch) on one line per logging step

### Import patterns

In training (`train/train_qwen_think.py`):

```python
from model_util.model_util import load_tokenizer
from model_util.file_util import ensure_dir, latest_valid_trainer_checkpoint
from model_util.train_util import (
    enforce_strict_and_length,
    get_map_and_tokenize_row,
    SingleLineMetricsCallback,
)
```

In inference (`infer/generate_qwen_vllm_think.py`):

```python
from model_util.file_util import ensure_dir, resolve_outfile, prepare_model_dir as resolve_model_dir
from model_util.text_util import normalize_text, extract_code_from_text
from model_util.model_util import get_model_config_fields, detect_model_type, load_tokenizer
```

### Notes

- Both `train/train_qwen_think.py` and `infer/generate_qwen_vllm_think.py` prepend the repository root to `sys.path` at runtime so `model_util` resolves without setting `PYTHONPATH`.
- `prepare_model_dir` requires an optional `ds_convert` (DeepSpeed ZeRO converter) when auto-converting ZeRO checkpoints; pass `None` if not available.


