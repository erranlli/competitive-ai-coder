# merge_shards_cpu.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
import shutil
import argparse

def parse_args():
    ap = argparse.ArgumentParser(description="Merge HF sharded checkpoint into a single-file directory (CPU-only)")
    ap.add_argument("--input-dir", required=True, help="Path to sharded model directory (containing shards/index)")
    ap.add_argument("--output-dir", required=True, help="Path to write single-file model directory")
    ap.add_argument("--max-shard-size", default="40GB", help="Max shard size used when saving (forces single file if large)")
    return ap.parse_args()

args = parse_args()
sharded_model_path = args.input_dir
single_file_output_path = args.output_dir

print("--- Starting Shard Consolidation to a Single File ---")

# Ensure the output directory exists
os.makedirs(single_file_output_path, exist_ok=True)

# 1. Load the model from the sharded checkpoint onto the CPU.
#    This will require a significant amount of RAM (~30 GB).
print(f"Loading sharded model from: {sharded_model_path}")

# Detect misplaced shard directory (e.g., a folder named 'pytorch_model.bin' containing the shards)
weights_dir = sharded_model_path
root_index_json = os.path.join(sharded_model_path, "pytorch_model.bin.index.json")
subdir = os.path.join(sharded_model_path, "pytorch_model.bin")
subdir_index_json = os.path.join(subdir, "pytorch_model.bin.index.json")

if not os.path.isfile(root_index_json) and os.path.isdir(subdir) and os.path.isfile(subdir_index_json):
    print("Detected shards under subdirectory 'pytorch_model.bin/'. Using that as weights path.")
    weights_dir = subdir

# Load config from the root directory (config is usually at the top-level)
config = AutoConfig.from_pretrained(sharded_model_path)

# We use device_map="cpu" to ensure no GPUs are used.
model = AutoModelForCausalLM.from_pretrained(
    weights_dir,
    config=config,
    dtype=torch.float32,        # Load in full precision for consolidation
    low_cpu_mem_usage=True,     # Helps manage memory during loading
    device_map="cpu",
)

# 2. Save the model to the new directory.
#    By setting max_shard_size to a large value, we force it to save as a single file.
print(f"Saving consolidated model to: {single_file_output_path}")
model.save_pretrained(
    single_file_output_path,
    max_shard_size=args.max_shard_size # Set higher than the model size to guarantee a single file
)

# 3. Load and save the tokenizer to the new directory.
print("Saving tokenizer and config files...")
tokenizer = AutoTokenizer.from_pretrained(sharded_model_path)
tokenizer.save_pretrained(single_file_output_path)


print("\n--- Consolidation Complete ---")
print(f"Your model is now ready in a single file at: {single_file_output_path}")
print("Use this new directory path with vLLM.")
