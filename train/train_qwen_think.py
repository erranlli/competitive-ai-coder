#!/usr/bin/env python3
# improved_finetune_qwen_codeforces.py
import argparse
import os
import sys
import gc
import math
import json
from typing import Any, Dict, List, Tuple, Optional
import glob
import shutil
from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict

import torch
from datasets import load_from_disk, Dataset

# Ensure repository root is on sys.path for local package imports
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_CUR_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from model_util.model_util import load_tokenizer
from model_util.file_util import ensure_dir, latest_valid_trainer_checkpoint
from model_util.train_util import (
    enforce_strict_format,
    enforce_length_only,
    get_map_and_tokenize_row,
    SingleLineMetricsCallback,
)
from trl import SFTTrainer
from tqdm import tqdm

# disable problematic deepspeed optimizations sometimes used on some infra
os.environ["DEEPSPEED_DISABLE_ZEROFLOW"] = "1"

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def apply_chat_template(tokenizer: AutoTokenizer, messages: List[Dict[str, str]]) -> str:
    # try built-in template if available; fallback to simple wrapper
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        parts: List[str] = []
        for m in messages:
            role = m.get('role', 'user').strip()
            content = m.get('content', '').strip()
            parts.append(f"<|{role}|>\n{content}\n")
        # add explicit solution marker expected by downstream masking
        return "\n".join(parts)

# --- Dataset loader + tokenizer mapping (with label masking) ---
def load_and_format_dataset(
    dataset_path: str,
    tokenizer: AutoTokenizer,
    validation_split_percentage: int,
    max_seq_length: int,
    enforce_format: bool = False,
    max_samples: int = 0,
    single_arrow_file: Optional[str] = None,
    use_first_arrow_in_dir: bool = False,
    enable_loss_reweight: bool = False,
    loss_weight_code: float = 1.0,
    loss_weight_noncode: float = 1.0,
    mask_noncode: bool = False,
) -> Tuple[Dataset, Dataset]:
    ds = None
    # Fast path: load a single Arrow shard
    if single_arrow_file and os.path.isfile(single_arrow_file):
        print(f"Loading dataset from single Arrow file: {single_arrow_file}")
        ds = Dataset.from_file(single_arrow_file)
    else:
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        # Optionally pick the first data-*.arrow inside the directory
        if use_first_arrow_in_dir and os.path.isdir(dataset_path):
            import glob as _glob
            arrow_files = sorted(_glob.glob(os.path.join(dataset_path, "data-*.arrow")))
            # Prefer data-00000-of-*.arrow if present for stability
            preferred = [p for p in arrow_files if os.path.basename(p).startswith("data-00000-of-")]
            pick = preferred[0] if preferred else (arrow_files[0] if arrow_files else None)
            if pick and os.path.isfile(pick):
                print(f"Loading dataset from first Arrow shard in dir: {pick}")
                ds = Dataset.from_file(pick)
        if ds is None:
            print(f"Loading dataset from local path: {dataset_path}")
            ds = load_from_disk(dataset_path)

    # support various split shapes
    if isinstance(ds, dict):
        if 'train' in ds and 'validation' in ds:
            train_ds, eval_ds = ds['train'], ds['validation']
        elif 'train' in ds and 'test' in ds:
            train_ds, eval_ds = ds['train'], ds['test']
        else:
            ds = next(iter(ds.values()))
            ds_splits = ds.train_test_split(test_size=float(validation_split_percentage) / 100.0, seed=42)
            train_ds, eval_ds = ds_splits["train"], ds_splits["test"]
    else:
        # Single file case or monolithic dataset: create a split
        ds_splits = ds.train_test_split(test_size=float(validation_split_percentage) / 100.0, seed=42)
        train_ds, eval_ds = ds_splits["train"], ds_splits["test"]

    print(f"Dataset split into {len(train_ds)} training samples and {len(eval_ds)} evaluation samples.")

    # Always enforce length; optionally enforce strict structure/pattern
    train_ds = enforce_length_only(train_ds, tokenizer, max_seq_length, 'train')
    eval_ds = enforce_length_only(eval_ds, tokenizer, max_seq_length, 'eval')
    if enforce_format:
        train_ds = enforce_strict_format(train_ds, tokenizer, max_seq_length, 'train')
        eval_ds = enforce_strict_format(eval_ds, tokenizer, max_seq_length, 'eval')


    map_and_tokenize_row = get_map_and_tokenize_row(
        tokenizer,
        max_seq_length,
        enable_loss_reweight,
        loss_weight_code,
        loss_weight_noncode,
        mask_noncode,
    )


    num_proc = min(os.cpu_count() - 2, 8) if os.cpu_count() > 4 else 1
    print(f"Using {num_proc} processes for dataset mapping and tokenization")

    train_mapped = train_ds.map(map_and_tokenize_row, remove_columns=train_ds.column_names, num_proc=num_proc)
    eval_mapped = eval_ds.map(map_and_tokenize_row, remove_columns=eval_ds.column_names, num_proc=num_proc)

    # filter short/empty
    initial_train_size, initial_eval_size = len(train_mapped), len(eval_mapped)
    train_mapped = train_mapped.filter(lambda ex: len(ex.get('input_ids', [])) > 16)
    eval_mapped = eval_mapped.filter(lambda ex: len(ex.get('input_ids', [])) > 16)

    print(f"Filtered out empty/short samples. Train: {initial_train_size} -> {len(train_mapped)}, "
          f"Eval: {initial_eval_size} -> {len(eval_mapped)}")

    if max_samples > 0:
        train_mapped = train_mapped.select(range(min(len(train_mapped), max_samples)))

    if len(train_mapped) == 0 or len(eval_mapped) == 0:
        raise ValueError("No valid training or evaluation samples found after filtering!")

    return train_mapped, eval_mapped


import os
import torch

def is_main_process():
    return int(os.environ.get("LOCAL_RANK", "0")) == 0

# (Removed PEFT/LoRA utilities)

def check_environment():
    print("=== Environment Check ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
    try:
        import deepspeed
        print(f"DeepSpeed version: {deepspeed.__version__}")
    except Exception as e:
        print(f"DeepSpeed import failed: {e}")
    try:
        from transformers import __version__ as tver
        print(f"Transformers version: {tver}")
    except Exception as e:
        print(f"Transformers import failed: {e}")
    print("=== Environment Check Complete ===\n")


def consolidate_deepspeed_checkpoint(trainer, tokenizer, output_dir: str):
    """
    Consolidates a raw DeepSpeed checkpoint into a standard, single-file
    Hugging Face model directory, making it compatible with vLLM.
    """
    if int(os.environ.get("LOCAL_RANK", "0")) != 0:
        return

    print("\n--- [Rank 0] Starting Final Model Consolidation ---")

    # Step 1: Find the latest checkpoint directory created by the Trainer.
    checkpoints = sorted(
        glob.glob(os.path.join(output_dir, "checkpoint-*")),
        key=lambda p: int(p.rsplit("-", 1)[-1])
    )
    if not checkpoints:
        print(f"FATAL ERROR: No checkpoint directory found in {output_dir}. Cannot consolidate.")
        return
    
    latest_checkpoint_dir = checkpoints[-1]
    print(f"Found latest checkpoint directory to convert: {latest_checkpoint_dir}")

    # Step 2: The DeepSpeed utility needs a "tag" which corresponds to the step number.
    # This is used to find the "global_stepX" folder inside the checkpoint dir.
    # Prefer 'latest' tag in the checkpoint dir; DS utility will read it if tag is None
    tag = None
    print("Using checkpoint tag from 'latest' file (DeepSpeed will auto-detect).")
    
    # Step 3: We will write weights into the output_dir (not a file path).
    print(f"Consolidated model weights will be saved under: {output_dir}")

    # Step 4: Run the DeepSpeed conversion utility. This is the critical step.
    print("Running DeepSpeed's conversion script...")
    try:
        # Run conversion: second arg must be a directory; utility writes files inside it
        convert_zero_checkpoint_to_fp32_state_dict(latest_checkpoint_dir, output_dir, tag=tag)
        print("DeepSpeed conversion successful!")
    except Exception as e:
        print(f"FATAL ERROR: DeepSpeed model conversion failed. Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 4b: If a leftover directory named 'pytorch_model.bin' exists (from an earlier buggy run),
    # move the inner file out and remove the directory so that vLLM finds a proper file.
    pm_path = os.path.join(output_dir, "pytorch_model.bin")
    if os.path.isdir(pm_path):
        inner_pm = os.path.join(pm_path, "pytorch_model.bin")
        inner_st = os.path.join(pm_path, "model.safetensors")
        try:
            if os.path.isfile(inner_pm):
                tmp_path = pm_path + ".tmpfile"
                shutil.move(inner_pm, tmp_path)
                shutil.rmtree(pm_path)
                shutil.move(tmp_path, pm_path)
                print("Fixed directory named 'pytorch_model.bin' -> replaced with real weight file.")
            elif os.path.isfile(inner_st):
                tmp_path = pm_path + ".tmpsf"
                shutil.move(inner_st, tmp_path)
                shutil.rmtree(pm_path)
                shutil.move(tmp_path, os.path.join(output_dir, "model.safetensors"))
                print("Fixed directory named 'pytorch_model.bin' -> moved safetensors to top-level.")
        except Exception as fix_e:
            print(f"Warning: failed to fix directory 'pytorch_model.bin': {fix_e}")

    # Step 5: Save the tokenizer and config files to the top-level output directory.
    print(f"Saving tokenizer and config to {output_dir}...")
    # Ensure required files exist for vLLM/HF
    try:
        trainer.model.config.save_pretrained(output_dir)
    except Exception as e:
        print(f"Warning: failed to save config.json: {e}")
    try:
        tokenizer.save_pretrained(output_dir)
    except Exception as e:
        print(f"Warning: failed to save tokenizer: {e}")

    print(f"--- Consolidation Complete ---")
    print(f"Final consolidated model saved to: {output_dir}")
    print("This directory is now ready for use with vLLM.")


# -------------------- Main --------------------
def main():
    check_environment()

    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--output-dir", default="./qwen2-7b-codeforces-lora")
    p.add_argument("--report-to", default="wandb", choices=["wandb", "none"])
    p.add_argument("--max-seq-length", type=int, default=32768) #
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--deepspeed", default="deepspeed_zero2.json")
    p.add_argument("--learning-rate", type=float, default=4e-5) #https://huggingface.co/blog/open-r1/update-3
    p.add_argument("--num-train-epochs", type=float, default=3.0)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--eval-steps", type=int, default=2000)
    p.add_argument("--save-steps", type=int, default=0)
    p.add_argument("--logging-steps", type=int, default=0)
    p.add_argument("--validation-split-percentage", type=int, default=5)
    p.add_argument("--early-stopping-patience", type=int, default=3)
    p.add_argument("--use-lora", action="store_true", default=False) #Turn Lora off
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--packing", action="store_true", default=False) #Packing hurts performance
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--wandb-project", default=None, help="(optional) wandb project name to report to")
    p.add_argument("--disable-training-eval", action="store_true", default=True, help="Disable Trainer evaluation during training to save time")
    p.add_argument("--single-arrow-file", type=str, default=None, help="Path to a single Arrow shard to load instead of the full dataset")
    p.add_argument("--enforce-format", action="store_true", default=False, help="If set, enforce strict message/format and length filters on the dataset")
    p.add_argument("--use-first-arrow-in-dir", action="store_true", default=False, help="If set, load the first data-*.arrow file under the dataset dir")
    p.add_argument("--resume-from-checkpoint", type=str, default=None, help="Path to a checkpoint dir to resume from (e.g., output_dir/checkpoint-123 or 'last')")
    # Loss re-weighting for code vs non-code tokens
    p.add_argument("--enable-loss-reweight", action="store_true", default=False, help="Enable per-token loss re-weighting (code vs non-code)")
    p.add_argument("--loss-weight-code", type=float, default=1.0, help="Loss weight for code tokens inside ```python blocks")
    p.add_argument("--loss-weight-noncode", type=float, default=1.0, help="Loss weight for non-code assistant tokens")
    p.add_argument("--mask-noncode", action="store_true", default=False, help="If set, non-code assistant tokens are masked from loss entirely")
    
    args = p.parse_args()

    if args.report_to == "wandb":
        os.environ["WANDB_PROJECT"] = "qwen-codeforces-finetune"

    ensure_dir(args.output_dir)

    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args.model_name)

    print("Loading and formatting dataset...")
    train_dataset, eval_dataset = load_and_format_dataset(
        args.dataset_path, tokenizer, args.validation_split_percentage,
        max_seq_length=args.max_seq_length, enforce_format=args.enforce_format, max_samples=args.max_train_samples,
        single_arrow_file=args.single_arrow_file, use_first_arrow_in_dir=args.use_first_arrow_in_dir,
        enable_loss_reweight=args.enable_loss_reweight,
        loss_weight_code=args.loss_weight_code,
        loss_weight_noncode=args.loss_weight_noncode,
        mask_noncode=args.mask_noncode
    )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, trust_remote_code=True,
        dtype=torch.bfloat16, use_cache=False, attn_implementation="flash_attention_2"
    )


    # TrainingArguments tuned for stable fine-tune
    do_eval = not args.disable_training_eval
    eval_strategy_value = "steps" if do_eval else "no"
    load_best_value = True if do_eval else False
    # Compute dynamic steps for ~0.5 epoch logging/eval and 2-epoch checkpointing
    try:
        world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    except Exception:
        world_size = 1
    updates_per_epoch = max(1, math.ceil(len(train_dataset) / (args.per_device_train_batch_size * world_size * args.gradient_accumulation_steps)))
    save_steps_calc = max(1, updates_per_epoch * 1)  # every 1 epochs
    log_steps_calc = max(1, updates_per_epoch // 20)  # ~ every 0.05 epoch

    # Build TrainingArguments with compatibility across Transformers versions
    ta_fields = getattr(TrainingArguments, "__dataclass_fields__", {})
    strategy_key = "evaluation_strategy" if "evaluation_strategy" in ta_fields else (
        "eval_strategy" if "eval_strategy" in ta_fields else None
    )

    ta_kwargs = dict(
        output_dir=args.output_dir,
        deepspeed=args.deepspeed if int(os.environ.get("WORLD_SIZE", "1")) > 1 else None,
        bf16=True,
        do_train=True,
        do_eval=do_eval,
        eval_steps=log_steps_calc,
        save_strategy="steps",
        save_steps=save_steps_calc,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        weight_decay=0.05, # For instruction/code SFT, common successful ranges are 0.01–0.05; TODO
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=log_steps_calc,
        save_total_limit=3,
        num_train_epochs=args.num_train_epochs,
        report_to=args.report_to,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        max_grad_norm=1.0, # curb exploding updates that drive verbosity and repetition
    )
    if strategy_key is not None:
        ta_kwargs[strategy_key] = eval_strategy_value
    if "load_best_model_at_end" in ta_fields:
        ta_kwargs["load_best_model_at_end"] = load_best_value
    if "metric_for_best_model" in ta_fields:
        ta_kwargs["metric_for_best_model"] = "eval_loss"
    if "greater_is_better" in ta_fields:
        ta_kwargs["greater_is_better"] = False

    training_args = TrainingArguments(**ta_kwargs)

    # Data collator that pads input_ids AND labels (label_pad_token_id=-100)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )

    class WeightedSFTTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            loss_weights = inputs.pop("loss_weights", None)
            outputs = model(**inputs)
            if loss_weights is None:
                # Compute and stash extra metrics; merge at on_log for single-line printing
                try:
                    # Robustly get logits across different ModelOutput types
                    logits = None
                    try:
                        logits = outputs.logits  # prefer attribute access
                    except Exception:
                        pass
                    if logits is None and isinstance(outputs, dict):
                        logits = outputs.get("logits", None)

                    labels = inputs.get("labels", None)
                    if logits is not None and labels is not None:
                        with torch.no_grad():
                            shift_logits = logits[..., :-1, :].contiguous()
                            shift_labels = labels[..., 1:].contiguous()
                            valid_mask = (shift_labels != -100)
                            if valid_mask.any():
                                pred_tokens = shift_logits.argmax(dim=-1)
                                correct = (pred_tokens == shift_labels) & valid_mask
                                mean_acc = (correct.sum().float() / valid_mask.sum().float()).item()
                                log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                                probs = torch.exp(log_probs)
                                ent = (-probs * log_probs).sum(dim=-1)
                                mean_ent = ent[valid_mask].mean().item()
                                if not hasattr(self, "_cumulative_tokens"):
                                    self._cumulative_tokens = 0.0
                                self._cumulative_tokens += float(valid_mask.sum().item())
                                self._last_extra_metrics = {
                                    "mean_token_accuracy": mean_acc,
                                    "entropy": mean_ent,
                                    "num_tokens": float(self._cumulative_tokens),
                                    "epoch": float(getattr(self.state, "epoch", 0.0) or 0.0),
                                }
                except Exception:
                    pass
                return (outputs.loss, outputs) if return_outputs else outputs.loss

            try:
                # Robustly get logits across different ModelOutput types
                logits = None
                try:
                    logits = outputs.logits
                except Exception:
                    pass
                if logits is None and isinstance(outputs, dict):
                    logits = outputs.get("logits", None)

                labels = inputs.get("labels", None)
                if logits is None or labels is None:
                    return (outputs.loss, outputs) if return_outputs else outputs.loss

                # Shift for next-token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                # Per-token CE loss (no reduction)
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                per_token_loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )

                # Convert and align weights to [B, T]
                B, T = labels.size(0), labels.size(1)
                device = labels.device
                dtype = per_token_loss.dtype

                w = loss_weights
                if not torch.is_tensor(w):
                    try:
                        w = torch.tensor(w, device=device, dtype=dtype)
                    except Exception:
                        w = None
                if w is None:
                    return (outputs.loss, outputs) if return_outputs else outputs.loss

                # Ensure 2D [B, T]
                if w.dim() == 1:
                    if w.numel() == T:
                        w = w.view(1, T).expand(B, T)
                    elif w.numel() == B * T:
                        w = w.view(B, T)
                    else:
                        w = w.new_ones((B, T))
                elif w.dim() == 2:
                    # Pad/trim rows/cols to [B,T]
                    if w.size(0) != B:
                        if w.size(0) > B:
                            w = w[:B, :]
                        else:
                            pad_rows = B - w.size(0)
                            w = torch.cat([w, w.new_ones((pad_rows, min(w.size(1), T)))], dim=0)
                    if w.size(1) != T:
                        if w.size(1) > T:
                            w = w[:, :T]
                        else:
                            pad_cols = T - w.size(1)
                            w = torch.cat([w, w.new_ones((B, pad_cols))], dim=1)
                else:
                    w = w.view(B, T)

                # Shift weights to align with shifted labels
                shifted_weights = w[..., 1:].contiguous()
                flat_weights = shifted_weights.view(-1)

                # Valid tokens mask (ignore -100)
                valid_mask = (shift_labels.view(-1) != -100)

                weighted_loss = per_token_loss * flat_weights
                sum_weighted = weighted_loss[valid_mask].sum()
                sum_weights = flat_weights[valid_mask].sum()
                loss = sum_weighted / torch.clamp(sum_weights, min=1.0)

                # Stash extra metrics for merging at log-time
                try:
                    with torch.no_grad():
                        valid_mask = (shift_labels != -100)
                        if valid_mask.any():
                            pred_tokens = shift_logits.argmax(dim=-1)
                            correct = (pred_tokens == shift_labels) & valid_mask
                            mean_acc = (correct.sum().float() / valid_mask.sum().float()).item()
                            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                            probs = torch.exp(log_probs)
                            ent = (-probs * log_probs).sum(dim=-1)
                            mean_ent = ent[valid_mask].mean().item()
                            if not hasattr(self, "_cumulative_tokens"):
                                self._cumulative_tokens = 0.0
                            self._cumulative_tokens += float(valid_mask.sum().item())
                            self._last_extra_metrics = {
                                "mean_token_accuracy": mean_acc,
                                "entropy": mean_ent,
                                "num_tokens": float(self._cumulative_tokens),
                                "epoch": float(getattr(self.state, "epoch", 0.0) or 0.0),
                            }
                except Exception:
                    pass

                return (loss, outputs) if return_outputs else loss
            except Exception:
                return (outputs.loss, outputs) if return_outputs else outputs.loss

    class SingleLineMetricsCallback(TrainerCallback):
        def __init__(self, trainer_ref):
            self.trainer_ref = trainer_ref
            self._last_grad_norm: Optional[float] = None

        def on_step_end(self, args, state, control, **kwargs):
            try:
                model = kwargs.get("model", None)
                if model is None and hasattr(self.trainer_ref, "model"):
                    model = self.trainer_ref.model
                grad_norm_sq = 0.0
                any_grad = False
                if model is not None:
                    for p in model.parameters():
                        if p.grad is not None:
                            g = p.grad.data
                            val = g.norm(2).item()
                            grad_norm_sq += val * val
                            any_grad = True
                grad_norm = (grad_norm_sq ** 0.5) if any_grad else 0.0

                # Prefer DeepSpeed's global grad norm if available
                try:
                    ds_engine = getattr(self.trainer_ref, "deepspeed", None)
                    if ds_engine is not None:
                        if hasattr(ds_engine, "get_global_grad_norm"):
                            gn = ds_engine.get_global_grad_norm()
                            if gn is not None:
                                grad_norm = float(gn)
                        elif hasattr(ds_engine, "get_global_norm"):
                            gn = ds_engine.get_global_norm()
                            if gn is not None:
                                grad_norm = float(gn)
                except Exception:
                    pass

                self._last_grad_norm = float(grad_norm)
            except Exception:
                self._last_grad_norm = None

        def on_log(self, args, state, control, logs=None, **kwargs):
            try:
                if logs is None:
                    return
                # Merge last extra metrics computed during compute_loss
                extra = getattr(self.trainer_ref, "_last_extra_metrics", None)
                if isinstance(extra, dict):
                    logs.update(extra)
                # Add grad_norm captured at step end so it prints on the same line
                if self._last_grad_norm is not None:
                    logs["grad_norm"] = self._last_grad_norm
            except Exception:
                return

    _callbacks: List[Any] = []
    if do_eval:
        _callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    trainer = WeightedSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=None,
        callbacks=_callbacks,
        data_collator=data_collator,
    )

    # Ensure single-line merged metrics at each logging step
    trainer.add_callback(SingleLineMetricsCallback(trainer))

    # Start training
    try:
        print("Starting training...")
        resume_arg = None
        if args.resume_from_checkpoint:
            req = args.resume_from_checkpoint.strip()
            chosen = None
            if req.lower() == "last":
                chosen = latest_valid_trainer_checkpoint(args.output_dir)
                if chosen:
                    print(f"Resuming from last valid checkpoint: {chosen}")
                else:
                    print("No valid checkpoints (with trainer_state.json) found; starting fresh.")
            else:
                cand = req
                if os.path.isdir(cand):
                    base = os.path.basename(cand)
                    if base.startswith("global_step"):
                        cand = os.path.dirname(cand)
                    if os.path.isfile(os.path.join(cand, "trainer_state.json")):
                        chosen = cand
                        print(f"Resuming from checkpoint: {chosen}")
                    else:
                        prev = latest_valid_trainer_checkpoint(os.path.dirname(cand))
                        if prev:
                            print(f"Requested checkpoint missing trainer_state.json. Falling back to previous valid: {prev}")
                            chosen = prev
                        else:
                            print(f"No valid checkpoints found near requested path: {req}. Starting fresh.")
                else:
                    print(f"Provided --resume-from-checkpoint does not exist: {req}. Starting fresh.")
            resume_arg = chosen

        trainer.train(resume_from_checkpoint=resume_arg)
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise

    # Wait for all processes to finish training
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # The trainer has already saved the sharded checkpoint due to `save_steps=1`.
    # Now, we just need to run our consolidation function on Rank 0.
    consolidate_deepspeed_checkpoint(trainer, tokenizer, args.output_dir)

    print("Script finished.")


if __name__ == "__main__":
    main()