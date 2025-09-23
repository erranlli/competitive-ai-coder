# consolidate_cpu.py
import os
import glob
import torch
import shutil
import argparse
from transformers import AutoTokenizer, AutoConfig

# IMPORTANT: This line tells PyTorch and DeepSpeed to not use any GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="CPU-only consolidation of DeepSpeed checkpoints into HF weights")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Training run directory containing checkpoint-*/ (ignored if --checkpoint-dir is provided)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Path to a specific checkpoint-*/ directory to consolidate (e.g., runs/exp/checkpoint-202)",
    )
    parser.add_argument(
        "--dest-dir",
        default=None,
        help="Destination directory to write consolidated fp32 weights + tokenizer/config. Defaults to --output-dir",
    )
    parser.add_argument(
        "--dest-filename",
        default=None,
        help="Destination filename for consolidated weights (default: pytorch_model.bin or pytorch_model_<step>.bin)",
    )
    parser.add_argument(
        "--base-model-name",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model name used to save tokenizer/config alongside consolidated weights",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite destination weights file if it exists",
    )
    args = parser.parse_args()

    run_dir = args.output_dir
    ckpt_dir = args.checkpoint_dir
    dest_dir = args.dest_dir or run_dir

    if ckpt_dir is None:
        # Find the latest checkpoint directory
        checkpoints = sorted(
            glob.glob(os.path.join(run_dir, "checkpoint-*")),
            key=lambda p: int(p.rsplit("-", 1)[-1]) if p.rsplit("-", 1)[-1].isdigit() else -1,
        )
        if not checkpoints:
            raise ValueError(f"FATAL ERROR: No checkpoint directory found in {run_dir}. Cannot consolidate.")
        ckpt_dir = checkpoints[-1]
    else:
        if not os.path.isdir(ckpt_dir):
            raise ValueError(f"Provided --checkpoint-dir does not exist or is not a directory: {ckpt_dir}")

    # Derive step suffix from checkpoint directory name
    step_suffix = None
    base = os.path.basename(os.path.normpath(ckpt_dir))
    if base.startswith("checkpoint-"):
        step_suffix = base.split("-", 1)[-1]

    ensure_dir(dest_dir)

    # Determine destination filename
    if args.dest_filename:
        dest_file = os.path.join(dest_dir, args.dest_filename)
    else:
        if step_suffix and step_suffix.isdigit():
            dest_file = os.path.join(dest_dir, f"pytorch_model_{step_suffix}.bin")
        else:
            dest_file = os.path.join(dest_dir, "pytorch_model.bin")

    # Avoid overwriting unless forced
    if os.path.exists(dest_file) and not args.force:
        # Append numeric suffix
        i = 1
        root, ext = os.path.splitext(dest_file)
        while os.path.exists(f"{root}.{i}{ext}"):
            i += 1
        dest_file = f"{root}.{i}{ext}"
        print(f"Destination exists, writing to: {dest_file}")

    print("--- Starting CPU-Only DeepSpeed Checkpoint Consolidation ---")
    print(f"Consolidating checkpoint: {ckpt_dir}")
    print(f"Writing fp32 weights to: {dest_file}")

    # 1) Run the DeepSpeed conversion utility to a single fp32 file
    print("Consolidating checkpoint... This may take a few minutes and require significant RAM.")
    try:
        convert_zero_checkpoint_to_fp32_state_dict(ckpt_dir, dest_file)
        print(f"Successfully consolidated checkpoint to '{dest_file}'")
    except Exception as e:
        print(f"FATAL ERROR: DeepSpeed model conversion failed. Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2) Save tokenizer and config files to the destination directory (for vLLM/HF)
    print(f"Saving tokenizer and config to {dest_dir}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
        tokenizer.save_pretrained(dest_dir)
        config = AutoConfig.from_pretrained(args.base_model_name)
        config.save_pretrained(dest_dir)
        print("Successfully saved tokenizer and config.")
    except Exception as e:
        print(f"Warning: Failed to save tokenizer/config: {e}")

    print("\n--- Consolidation Complete ---")
    print(f"Final consolidated model is ready in: {dest_dir}")
    print("This directory can now be used with vLLM.")

if __name__ == "__main__":
    main()

