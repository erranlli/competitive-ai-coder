# consolidate_cpu.py
import os
import glob
import torch
import shutil
from transformers import AutoTokenizer, AutoConfig

# IMPORTANT: This line tells PyTorch and DeepSpeed to not use any GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict

def main():
    # --- CONFIGURE THESE PATHS ---
    # This should be the main output directory that contains the 'checkpoint-xxx' folder(s).
    output_dir = "/root/competitive-coding-ai/deepcoder-run2-z3-optimized"
    
    # This is the original base model you used for fine-tuning.
    base_model_name = "Qwen/Qwen2.5-7B-Instruct"
    # --- END OF CONFIGURATION ---

    print("--- Starting CPU-Only DeepSpeed Checkpoint Consolidation ---")

    # 1. Find the latest checkpoint directory
    checkpoints = sorted(
        glob.glob(os.path.join(output_dir, "checkpoint-*")),
        key=lambda p: int(p.rsplit("-", 1)[-1])
    )
    if not checkpoints:
        raise ValueError(f"FATAL ERROR: No checkpoint directory found in {output_dir}. Cannot consolidate.")
    
    latest_checkpoint_dir = checkpoints[-1]
    print(f"Found latest checkpoint to convert: {latest_checkpoint_dir}")

    # 2. Run the DeepSpeed conversion utility.
    # This will create a 'pytorch_model.bin' file inside the main output_dir.
    print("Consolidating checkpoint... This may take a few minutes and require significant RAM.")
    try:
        # The utility will automatically find the 'global_stepXXX' folder inside the checkpoint dir.
        convert_zero_checkpoint_to_fp32_state_dict(latest_checkpoint_dir, os.path.join(output_dir, "pytorch_model.bin"))
        print("Successfully consolidated checkpoint to 'pytorch_model.bin'")
    except Exception as e:
        print(f"FATAL ERROR: DeepSpeed model conversion failed. Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Save the tokenizer and config files to the top-level output directory.
    # A complete model for vLLM needs these files.
    print(f"Saving tokenizer and config to {output_dir}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.save_pretrained(output_dir)
        
        config = AutoConfig.from_pretrained(base_model_name)
        config.save_pretrained(output_dir)
        print("Successfully saved tokenizer and config.")
    except Exception as e:
        print(f"Warning: Failed to save tokenizer/config: {e}")

    print("\n--- Consolidation Complete ---")
    print(f"Final consolidated model is ready for inference in: {output_dir}")
    print("This directory can now be used with vLLM.")

if __name__ == "__main__":
    main()

