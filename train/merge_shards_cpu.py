# merge_shards_cpu.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
import shutil
import argparse

def parse_args():
    ap = argparse.ArgumentParser(description="Merge HF sharded (or unify single-file) checkpoint into a single-file directory (CPU-only)")
    ap.add_argument("--input-dir", "--input", dest="input_dir", required=True, help="Path to model directory (sharded or single-file)")
    ap.add_argument("--output-dir", "--output", dest="output_dir", required=True, help="Path to write single-file model directory")
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

# Fast-path: directory already contains a single weight file (but maybe non-standard name)
def copy_if_single_file(src_dir: str, dst_dir: str) -> bool:
    """If src_dir already has a single .bin/.safetensors weight file, copy/normalize it and tokenizer/config.
    Returns True if handled, False otherwise.
    """
    # Preferred names
    std_bin = os.path.join(src_dir, "pytorch_model.bin")
    std_safe = os.path.join(src_dir, "model.safetensors")
    os.makedirs(dst_dir, exist_ok=True)
    copied = False
    if os.path.isfile(std_bin):
        shutil.copy2(std_bin, os.path.join(dst_dir, "pytorch_model.bin"))
        copied = True
    elif os.path.isfile(std_safe):
        shutil.copy2(std_safe, os.path.join(dst_dir, "model.safetensors"))
        copied = True
    else:
        # Look for a likely consolidated bin (e.g., pytorch_model_202.bin)
        cand = None
        for name in sorted(os.listdir(src_dir)):
            if name.endswith(".bin") and name.startswith("pytorch_model_"):
                cand = os.path.join(src_dir, name)
                break
        if cand and os.path.isfile(cand):
            shutil.copy2(cand, os.path.join(dst_dir, "pytorch_model.bin"))
            copied = True

    if copied:
        # Copy common config/tokenizer artifacts if present
        for fname in [
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "added_tokens.json",
            "chat_template.jinja",
        ]:
            src = os.path.join(src_dir, fname)
            if os.path.isfile(src):
                try:
                    shutil.copy2(src, os.path.join(dst_dir, fname))
                except Exception:
                    pass
        print(f"Detected existing single-file weights in '{src_dir}'. Copied/normalized into '{dst_dir}'.")
        return True
    return False

# If already single-file, just copy/normalize and exit
root_index_json = os.path.join(weights_dir, "pytorch_model.bin.index.json")
if not os.path.isfile(root_index_json):
    handled = copy_if_single_file(weights_dir, single_file_output_path)
    if handled:
        print("Done (no merge needed).")
        exit(0)

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
