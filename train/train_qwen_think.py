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
from peft import LoraConfig, PeftModel
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
from peft import PeftModel

def is_main_process():
    return int(os.environ.get("LOCAL_RANK", "0")) == 0

def merge_and_eval_adapter(adapter_dir: str, base_model_name: str, tokenizer, eval_prompts: List[str], eval_refs: List[str], out_prefix: str):
    """
    adapter_dir: path where LoRA adapter (Peft) is saved
    base_model_name: original HF model id (e.g. Qwen/Qwen2.5-7B-Instruct)
    tokenizer: tokenizer instance (padding_side already set)
    eval_prompts / eval_refs: lists extracted from eval dataset (strings)
    out_prefix: prefix for merged model directory
    """

    if not is_main_process():
        print("Not main process; skipping merge/eval.")
        return

    # If adapter_folder looks like a deepspeed consolidated checkpoint, abort and explain
    # A PEFT adapter saved via trainer.save_model should contain adapter_config.json and pytorch_model.bin or adapter_model.bin
    # but DeepSpeed Zero-3 consolidated checkpoint folders can look different.
    adapter_files = os.listdir(adapter_dir)
    # quick check
    if any(f.startswith("global_step") or f.endswith(".pt") for f in adapter_files) and "adapter_config.json" not in adapter_files:
        print("ERROR: the adapter directory looks like a DeepSpeed ZeRO-3 checkpoint or a non-PEFT folder.")
        print("If you trained with DeepSpeed ZeRO-3, please run the DeepSpeed zero_to_fp32 conversion (see instructions).")
        print("Alternatively, ensure you saved the adapter via `trainer.save_model(adapter_dir)` which writes a PEFT adapter layout.")
        raise RuntimeError("Adapter directory not in PEFT format. Cannot merge safely.")

    print(f"Loading clean base model from hub: {base_model_name} (cpu, fp32) for merging...")
    # Load the *original* base model from the hub (not any ds checkpoint) — load on CPU to avoid memory pressure.
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    print("Loading PEFT adapter into base model...")
    peft_loaded = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=False)

    # find missing keys (PEFT prints some warnings, but we check explicitly)
    # `peft_loaded` has merged params but we can inspect expected vs found keys using its state_dict
    adapter_state = {}
    try:
        adapter_state = peft_loaded.peft_config.__dict__  # placeholder; we mostly rely on PeftModel errors below
    except Exception:
        pass

    # Merge and unload: this will create the merged weights in memory
    print("Merging adapter weights into the base model (this may take a while)...")
    try:
        merged = peft_loaded.merge_and_unload()
    except Exception as e:
        print("ERROR during merge_and_unload(). This may happen if the adapter doesn't match the base model.")
        raise

    merged_model_path = f"{out_prefix}-merged"
    print(f"Saving merged model to: {merged_model_path}")
    merged.save_pretrained(merged_model_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_model_path)
    print("Merged model saved.")

    # Move merged model to GPU for evaluation if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading merged model to {device} for evaluation...")
    merged_for_eval = AutoModelForCausalLM.from_pretrained(merged_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    merged_for_eval.eval()

    # Greedy generation/eval (small batches)
    print("Running greedy generation for PASS@1 evaluation...")
    batch_size = 2
    gens = []
    for i in range(0, len(eval_prompts), batch_size):
        batch = eval_prompts[i:i+batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            out = merged_for_eval.generate(
                **enc,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.0,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        for seq in out:
            gens.append(tokenizer.decode(seq, skip_special_tokens=True))
    # basic normalized string match metric (you can replace with judge-run)
    def normalize(s):
        import re
        s = s.strip()
        s = re.sub(r'\r\n', '\n', s)
        s = re.sub(r'\s+\n', '\n', s)
        return s
    correct = 0
    for ref, gen in zip(eval_refs, gens):
        if normalize(ref) == normalize(gen) or normalize(ref) in normalize(gen):
            correct += 1
    pass1 = correct / max(1, len(eval_prompts))
    print(f"PASS@1 (normalized-string-match) = {pass1*100:.2f}%")

    return merged_model_path

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
    p.add_argument("--save-steps", type=int, default=400)
    p.add_argument("--logging-steps", type=int, default=20)
    p.add_argument("--validation-split-percentage", type=int, default=5)
    p.add_argument("--early-stopping-patience", type=int, default=3)
    p.add_argument("--use-lora", action="store_true", default=False) #Turn Lora off
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--lora-dropout", type=float, default=0.05)
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
        torch_dtype=torch.bfloat16, use_cache=False, attn_implementation="flash_attention_2"
    )

    peft_config = None
    if args.use_lora:
        print("Configuring LoRA...")
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

    # TrainingArguments tuned for stable fine-tune
    do_eval = not args.disable_training_eval
    eval_strategy_value = "steps" if do_eval else "no"
    load_best_value = True if do_eval else False
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        deepspeed=args.deepspeed if int(os.environ.get("WORLD_SIZE", "1")) > 1 else None,
        bf16=True, do_train=True, do_eval=do_eval,
        eval_strategy=eval_strategy_value, eval_steps=args.eval_steps,
        save_strategy="steps", save_steps=args.save_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate, weight_decay=0.1,
        warmup_ratio=args.warmup_ratio, lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_total_limit=3,
        load_best_model_at_end=False, #TODO load_best_value, metric_for_best_model="eval_loss", greater_is_better=False,
        num_train_epochs=args.num_train_epochs,
        report_to=args.report_to,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

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
        peft_config=peft_config,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)] if do_eval else None,
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
