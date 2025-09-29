#!/usr/bin/env python3
# improved_finetune_qwen_codeforces_instrumented.py
import argparse
import os
import gc
import math
import json
from typing import Any, Dict, List, Tuple, Optional
import glob
import shutil
from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict

import torch
from datasets import load_dataset, load_from_disk, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig
from trl import SFTTrainer
import numpy as np
import re
from tqdm import tqdm

# disable problematic deepspeed optimizations sometimes used on some infra
os.environ["DEEPSPEED_DISABLE_ZEROFLOW"] = "1"

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# --- Utilities to build messages / extract solution ---
def build_messages_from_row(row: Dict[str, Any]) -> List[Dict[str, str]]:
    if isinstance(row.get("messages"), list) and row["messages"]:
        msgs = []
        for m in row["messages"]:
            role = m.get("role") or m.get("from") or "user"
            content = m.get("content") or m.get("value") or ""
            if content is None: content = ""
            msgs.append({"role": role, "content": str(content)})
        return msgs
    # fallback: try common fields
    if "input" in row and "output" in row:
        return [{"role": "user", "content": str(row["input"])},
                {"role": "assistant", "content": str(row["output"])}]
    return []


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
    max_seq_length: int,
    validation_split_percentage: int,
    max_samples: int = 0,
) -> Tuple[Dataset, Dataset]:
    """
    Loads, splits, filters by token length, sub-samples, and tokenizes the dataset.
    """
    print(f"Loading dataset from local path: {dataset_path}")
    ds = load_dataset("arrow", data_files=os.path.join(dataset_path, "*.arrow"))

    # 1. Handle dataset splitting
    if isinstance(ds, dict):
        train_ds, eval_ds = ds.get('train'), ds.get('validation')
        if not train_ds or not eval_ds:
            ds = next(iter(ds.values()))
            ds_splits = ds.train_test_split(test_size=validation_split_percentage / 100.0, seed=42)
            train_ds, eval_ds = ds_splits["train"], ds_splits["test"]
    else:
        ds_splits = ds.train_test_split(test_size=validation_split_percentage / 100.0, seed=42)
        train_ds, eval_ds = ds_splits["train"], ds_splits["test"]

    print(f"Initial dataset split: {len(train_ds)} training samples, {len(eval_ds)} evaluation samples.")

    # 2. Define the filtering function
    def filter_by_length(dataset_split: Dataset, split_name: str) -> Dataset:
        initial_count = len(dataset_split)
        if initial_count == 0:
            return dataset_split

        filtered_indices = []
        print(f"Filtering {split_name} dataset by token length (<= {max_seq_length})...")

        for i, row in enumerate(tqdm(dataset_split, desc=f"Scanning {split_name} for length")):
            messages = build_messages_from_row(row)
            if not (len(messages) == 2 and messages[0]['role'] == 'user'):
                continue

            try:
                full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                tokenized = tokenizer(full_text, add_special_tokens=False)
                if 16 < len(tokenized["input_ids"]) <= max_seq_length:
                    filtered_indices.append(i)
            except Exception:
                continue

        filtered_ds = dataset_split.select(filtered_indices)
        print(f"Length filter for {split_name}: Kept {len(filtered_ds)} of {initial_count} samples.")
        return filtered_ds

    # 3. Apply the length filter *before* mapping
    train_ds = filter_by_length(train_ds, "train")
    eval_ds = filter_by_length(eval_ds, "eval")

    if len(train_ds) == 0:
        raise ValueError("No valid training samples found after length filtering!")

    # --- max_samples block ---
    if max_samples > 0:
        print(f"Subsampling training data from {len(train_ds)} down to {max_samples} samples.")
        train_ds = train_ds.select(range(min(len(train_ds), max_samples)))

    if len(eval_ds) == 0:
        print("Warning: No evaluation samples found after length filtering. Creating a small dummy eval set from train set.")
        if len(train_ds) > 10:
             eval_ds = train_ds.select(range(10))
        else:
             raise ValueError("Not enough training samples to create a dummy evaluation set.")

    # 5. Define the mapping function for tokenization
    def map_and_tokenize_row(row: Dict[str, Any]) -> Dict[str, Any]:
        messages = build_messages_from_row(row)

        prompt_text = tokenizer.apply_chat_template([messages[0]], tokenize=False, add_generation_prompt=True)
        prompt_length = len(tokenizer(prompt_text, add_special_tokens=False)['input_ids'])

        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        tokenized = tokenizer(
            full_text,
            truncation=True,
            padding=False,
            max_length=max_seq_length,
            add_special_tokens=False,
        )

        labels = list(tokenized["input_ids"])
        labels[:prompt_length] = [-100] * prompt_length
        tokenized["labels"] = labels
        return tokenized

    # 6. Map the tokenization function over the filtered and sampled data
    num_proc = min(os.cpu_count() - 2, 8) if os.cpu_count() > 4 else 1
    print(f"Using {num_proc} processes for final dataset mapping and tokenization.")

    train_mapped = train_ds.map(map_and_tokenize_row, remove_columns=train_ds.column_names, num_proc=num_proc)
    eval_mapped = eval_ds.map(map_and_tokenize_row, remove_columns=eval_ds.column_names, num_proc=num_proc)

    return train_mapped, eval_mapped


def is_main_process():
    return int(os.environ.get("LOCAL_RANK", "0")) == 0


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

    checkpoints = sorted(
        glob.glob(os.path.join(output_dir, "checkpoint-*")),
        key=lambda p: int(p.rsplit("-", 1)[-1])
    )
    if not checkpoints:
        print(f"FATAL ERROR: No checkpoint directory found in {output_dir}. Cannot consolidate.")
        return

    latest_checkpoint_dir = checkpoints[-1]
    print(f"Found latest checkpoint directory to convert: {latest_checkpoint_dir}")

    tag = None
    print("Using checkpoint tag from 'latest' file (DeepSpeed will auto-detect).")
    print(f"Consolidated model weights will be saved under: {output_dir}")
    print("Running DeepSpeed's conversion script...")
    try:
        convert_zero_checkpoint_to_fp32_state_dict(latest_checkpoint_dir, output_dir, tag=tag)
        print("DeepSpeed conversion successful!")
    except Exception as e:
        print(f"FATAL ERROR: DeepSpeed model conversion failed. Error: {e}")
        import traceback
        traceback.print_exc()
        return

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

    print(f"Saving tokenizer and config to {output_dir}...")
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

    p = argparse.ArgumentParser(description="Fine-tune a Qwen model for competitive programming.")
    # --- Core Paths ---
    p.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--output-dir", required=True)

    # --- Training Hyperparameters ---
    p.add_argument("--num-train-epochs", type=float, default=3.0)
    p.add_argument("--learning-rate", type=float, default=4e-5)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--warmup-ratio", type=float, default=0.03)

    # --- Batching and Sequence Length ---
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--max-seq-length", type=int, default=16384)

    # --- Checkpointing and Logging ---
    p.add_argument("--save-strategy", default="epoch", choices=["steps", "epoch"], help="Save checkpoints by 'epoch' or 'steps'.")
    p.add_argument("--save-steps", type=int, default=0, help="If saving by steps, the step interval. If saving by epoch, this is ignored.")
    p.add_argument("--save-total-limit", type=int, default=50)
    p.add_argument("--logging-steps", type=int, default=10)

    # --- System and DeepSpeed ---
    p.add_argument("--deepspeed", type=str, default=None, help="Path to DeepSpeed config file.")
    p.add_argument("--bf16_full_eval", action="store_true", help="Enable full evaluation in bf16 to save memory.")
    p.add_argument("--report-to", default="wandb", choices=["wandb", "none"])
    p.add_argument("--wandb-project", default="new-ft-qwen2.5-mot", help="WandB project name.")

    # --- Data Handling ---
    p.add_argument("--validation-split-percentage", type=int, default=5)
    p.add_argument("--max-train-samples", type=int, default=0, help="If set, subsample the training set.")

    # --- Weight decay / resume control ---
    p.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay to pass into TrainingArguments (important).")
    p.add_argument("--resume-from-checkpoint", type=str, default=None, help="Path to checkpoint to resume from.")
    p.add_argument("--force-fresh-start", action="store_true", help="If set ignore --resume-from-checkpoint and start fresh (useful when changing optimizer hyperparams).")

    args = p.parse_args()

    # if user asked to force fresh start, ignore resume
    if args.force_fresh_start and args.resume_from_checkpoint:
        print(f"WARNING: --force-fresh-start set; ignoring resume-from-checkpoint {args.resume_from_checkpoint}")
        args.resume_from_checkpoint = None

    # --- Setup ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    train_dataset, eval_dataset = load_and_format_dataset(
        dataset_path=args.dataset_path,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        validation_split_percentage=args.validation_split_percentage,
        max_samples=args.max_train_samples,
    )

    # quick sanity print of tokenized samples (helpful for diffs)
    try:
        print("=== sample tokenized input ids (first 3 training examples) ===")
        for i in range(min(3, len(train_dataset))):
            s = train_dataset[i]
            print(f"sample {i}: input_ids_len={len(s.get('input_ids',[]))} labels_len={len(s.get('labels',[]))}")
    except Exception as e:
        print("Warning: could not print tokenized samples:", e)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        attn_implementation="flash_attention_2"
    )

    # --- Correctly handle save_steps logic ---
    save_steps_value = args.save_steps
    if args.save_strategy == "epoch":
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        updates_per_epoch = math.ceil(len(train_dataset) / (args.per_device_train_batch_size * world_size * args.gradient_accumulation_steps))
        save_steps_value = updates_per_epoch
        print(f"Save strategy is 'epoch', will save every {save_steps_value} steps.")

    # --- Training arguments (note weight_decay passed) ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        deepspeed=args.deepspeed,
        bf16=True,
        bf16_full_eval=args.bf16_full_eval,
        do_train=True,
        do_eval=False,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=save_steps_value,
        save_total_limit=args.save_total_limit,
        report_to=args.report_to,
        remove_unused_columns=False,
        weight_decay=args.weight_decay,
    )

    # debug print
    print("=== TrainingArguments summary (selected) ===")
    print("weight_decay:", training_args.weight_decay)
    print("num_train_epochs:", training_args.num_train_epochs)
    print("per_device_train_batch_size:", training_args.per_device_train_batch_size)
    print("gradient_accumulation_steps:", training_args.gradient_accumulation_steps)
    print("max_seq_length (tokenization):", args.max_seq_length)
    if args.deepspeed and os.path.exists(args.deepspeed):
        try:
            with open(args.deepspeed, "r") as f:
                ds_json = json.load(f)
            print("DeepSpeed JSON (optimizer section if present):", ds_json.get("optimizer"))
        except Exception as e:
            print("Warning: could not read deepspeed json:", e)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Try to eagerly create optimizer so we can inspect param groups (may be no-op in some trainer variants)
    try:
        # Many Trainer implementations expose create_optimizer_and_scheduler
        if hasattr(trainer, "create_optimizer_and_scheduler"):
            trainer.create_optimizer_and_scheduler()
        elif hasattr(trainer, "create_optimizer"):
            trainer.create_optimizer()
    except Exception as e:
        print("Note: could not pre-create optimizer (Trainer may create it at train start):", e)

    # Inspect optimizer param-groups
    try:
        optim = getattr(trainer, "optimizer", None)
        if optim is None:
            # sometimes the optimizer is stored under trainer.optimizer
            optim = getattr(trainer, "_optimizer", None)
        if optim is not None:
            wds = list({float(pg.get("weight_decay", 0.0)) for pg in optim.param_groups})
            print(">>> optimizer param-group weight_decay values:", wds)
        else:
            print(">>> trainer.optimizer is not available yet (will be created at train())")
    except Exception as e:
        print("Warning: failed to inspect optimizer param groups:", e)

    # Warn about resume-from-checkpoint implications
    if args.resume_from_checkpoint:
        print(f"WARNING: Resuming from checkpoint: {args.resume_from_checkpoint}")
        print("If the checkpoint contains optimizer state trained with a different weight_decay, the optimizer state may preserve that configuration.")
        print("For a clean test of new weight_decay, re-run without --resume-from-checkpoint or use --force-fresh-start to ignore resume.")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    consolidate_deepspeed_checkpoint(trainer.model, tokenizer, args.output_dir)


if __name__ == "__main__":
    main()
