#!/usr/bin/env python3
# improved_finetune_qwen_codeforces.py
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
from datasets import load_from_disk, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
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
    validation_split_percentage: int,
    max_seq_length: int,
    max_samples: int = 0,
    single_arrow_file: Optional[str] = None,
    use_first_arrow_in_dir: bool = False,
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

    # --- Enforce strict formatting and normalize constraints ---
    def _extract_messages(row: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        msgs = build_messages_from_row(row)
        if not (len(msgs) == 2 and msgs[0]['role'] == 'user' and msgs[1]['role'] == 'assistant'):
            return None, None
        return str(msgs[0]['content'] or ''), str(msgs[1]['content'] or '')


    # This pattern is based on the previous "flexible_pattern"
    strict_pattern = re.compile(
        # The string must start here, with optional whitespace
        r"^\s*"
        
        # 1. A required <think> block
        r"<think>[\s\S]+?</think>"
        
        # 2. FLEXIBLE SEPARATOR: Allow any characters between sections
        r"[\s\S]*?"
        
        # 3. A required 'Approach' section
        r"###\s*Approach"
        r"[\s\S]+?"
        
        # 4. An OPTIONAL 'Solution Code' header.
        # The non-capturing group (?:...) combined with the ? quantifier
        # makes this entire sub-pattern optional (matches zero or one time).
        r"(?:###\s*Solution Code\s*)?"
        
        # 5. A required python code block
        r"```python[\r\n]+"
        r"[\s\S]+?"
        r"```"
        
        # 6. FLEXIBLE SEPARATOR: Allow any characters before the explanation
        r"[\s\S]*?"
        r"###\s*Explanation"
        r"[\s\S]*"
        
        # The string must end here
        r"\Z",
        
        # Flag to make the pattern case-insensitive
        re.IGNORECASE,
    )

    def _enforce_on_split(ds_in: Dataset, split_name: str) -> Dataset:
        invalid = 0
        too_long = 0
        accepted: List[Dict[str, Any]] = []
        for row in ds_in:
            user_text, assistant_text = _extract_messages(row)
            if user_text is None or assistant_text is None:
                invalid += 1
                rid = row.get('id') or f"{row.get('contest_id','')}/{row.get('index','')}"
                print(f"[STRICT] {split_name} invalid (bad message shape) id={rid}")
                continue
            # Add default constraints if missing: insert immediately after Problem and before Input Format
            if '## Constraints' not in user_text:
                constraints_block = "\n\n## Constraints\nTime limit per test: 2.0 seconds\nMemory limit per test: 256.0 megabytes"
                m_input = re.search(r'(^|\n)##\s*Input\s*Format', user_text, re.IGNORECASE)
                if m_input:
                    user_text = user_text[:m_input.start()] + constraints_block + user_text[m_input.start():]
                else:
                    m_problem = re.search(r'(^|\n)#\s*Problem[^\n]*\n', user_text, re.IGNORECASE)
                    if m_problem:
                        insert_pos = m_problem.end()
                        user_text = user_text[:insert_pos] + constraints_block + user_text[insert_pos:]
                    else:
                        user_text = user_text.strip() + constraints_block
            # Validate assistant formatting
            if not strict_pattern.search(assistant_text):
                invalid += 1
                rid = row.get('id') or f"{row.get('contest_id','')}/{row.get('index','')}"
                print(f"[STRICT] {split_name} invalid format id={rid}")
                try:
                    print((assistant_text or '')[:400])
                except Exception:
                    pass
                continue
            # Token length filter: combined user + assistant under max_seq_length
            messages_pair = [
                {'role': 'user', 'content': user_text},
                {'role': 'assistant', 'content': assistant_text},
            ]
            try:
                full_text = tokenizer.apply_chat_template(messages_pair, tokenize=False, add_generation_prompt=False)
                tok = tokenizer(full_text, add_special_tokens=False)
                total_tokens = len(tok.get('input_ids', []))
            except Exception:
                total_tokens = 10**9  # force skip on tokenizer error
            if total_tokens > max_seq_length:
                too_long += 1
                rid = row.get('id') or f"{row.get('contest_id','')}/{row.get('index','')}"
                print(f"[LENGTH] {split_name} skip id={rid} tokens={total_tokens} > limit={max_seq_length}")
                continue

            # Write back normalized fields
            new_row = dict(row)
            if 'messages' in row and isinstance(row['messages'], list):
                new_row['messages'] = messages_pair
            else:
                # fallback to input/output
                new_row['input'] = user_text
                new_row['output'] = assistant_text
            accepted.append(new_row)
        print(f"[STRICT] {split_name} rejected {invalid} samples; [LENGTH] skipped {too_long}; kept {len(accepted)}")
        return Dataset.from_list(accepted) if accepted else Dataset.from_list([])

    train_ds = _enforce_on_split(train_ds, 'train')
    eval_ds = _enforce_on_split(eval_ds, 'eval')


    def map_and_tokenize_row(row: Dict[str, Any]) -> Dict[str, Any]:
        messages = build_messages_from_row(row)
        
        # Ensure the conversation is a valid single-turn exchange
        if not (len(messages) == 2 and messages[0]['role'] == 'user' and messages[1]['role'] == 'assistant'):
            return {"input_ids": [], "labels": []}

        # 1. Create the prompt text by applying the template to the user message only.
        #    `add_generation_prompt=True` tells the tokenizer to add the special tokens
        #    that cue the assistant's response (e.g., for Qwen2, it adds `<|im_start|>assistant\n`).
        #    This gives us the exact prefix that needs to be masked.
        prompt_text = tokenizer.apply_chat_template(
            [messages[0]],  # Only the user message
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 2. Create the full conversation text, which includes the assistant's solution.
        full_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )

        # 3. Tokenize the prompt text to find its length.
        prompt_token_ids = tokenizer(prompt_text, add_special_tokens=False)['input_ids']
        prompt_length = len(prompt_token_ids)

        # 4. Tokenize the full text. This is what the model will actually see.
        tokenized = tokenizer(
            full_text,
            truncation=True,
            padding=False,
            max_length=max_seq_length,
            add_special_tokens=False,
        )

        # 5. Create the labels and mask the prompt section.
        #    The first `prompt_length` tokens correspond to the user message and the assistant cue.
        labels = tokenized["input_ids"].copy()
        labels[:prompt_length] = [-100] * prompt_length
        
        tokenized["labels"] = labels
        return tokenized


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
    p.add_argument("--use-first-arrow-in-dir", action="store_true", default=False, help="If set, load the first data-*.arrow file under the dataset dir")
    p.add_argument("--resume-from-checkpoint", type=str, default=None, help="Path to a checkpoint dir to resume from (e.g., output_dir/checkpoint-123 or 'last')")
    
    args = p.parse_args()

    if args.report_to == "wandb":
        os.environ["WANDB_PROJECT"] = "qwen-codeforces-finetune"

    ensure_dir(args.output_dir)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # IMPORTANT: decoder-only models require left padding for correct generation
    tokenizer.padding_side = "left"

    print("Loading and formatting dataset...")
    train_dataset, eval_dataset = load_and_format_dataset(
        args.dataset_path, tokenizer, args.validation_split_percentage,
        max_seq_length=args.max_seq_length, max_samples=args.max_train_samples,
        single_arrow_file=args.single_arrow_file, use_first_arrow_in_dir=args.use_first_arrow_in_dir
    )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, trust_remote_code=True,
        dtype=torch.bfloat16, use_cache=False, attn_implementation="flash_attention_2"
    )

    # (Removed LoRA/PEFT configuration)

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
    save_steps_calc = max(1, updates_per_epoch * 2)  # every 2 epochs
    log_steps_calc = max(1, updates_per_epoch // 2)  # ~ every 0.5 epoch

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
        weight_decay=0.1,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=log_steps_calc,
        save_total_limit=3,
        num_train_epochs=args.num_train_epochs,
        report_to=args.report_to,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
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

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=None,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)] if do_eval else None,
        packing=args.packing,
        max_seq_length=args.max_seq_length,
        data_collator=data_collator,
    )

    # Start training
    try:
        print("Starting training...")
        resume_arg = None
        if args.resume_from_checkpoint:
            if args.resume_from_checkpoint.strip().lower() == "last":
                # Find latest checkpoint-* under output_dir
                ckpts = sorted(glob.glob(os.path.join(args.output_dir, "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[-1]) if p.rsplit("-", 1)[-1].isdigit() else -1)
                if ckpts:
                    resume_arg = ckpts[-1]
                    print(f"Resuming from last checkpoint: {resume_arg}")
                else:
                    print("No checkpoint-* directories found to resume from; starting fresh.")
            else:
                if os.path.isdir(args.resume_from_checkpoint):
                    resume_arg = args.resume_from_checkpoint
                    print(f"Resuming from checkpoint: {resume_arg}")
                else:
                    print(f"Provided --resume-from-checkpoint does not exist: {args.resume_from_checkpoint}. Starting fresh.")

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