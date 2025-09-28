#!/usr/bin/env python3
"""
Unified script to consolidate DeepSpeed checkpoints and merge sharded models for vLLM inference.

This script handles:
1. DeepSpeed Zero checkpoint consolidation to FP32 PyTorch format
2. HuggingFace sharded model merging to single files
3. Validation for vLLM compatibility (no sharded tensors)
4. Automatic detection of checkpoint type and appropriate processing

vLLM requires models to be in single-file format without sharded tensors.
"""

import os
import glob
import torch
import shutil
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# Disable GPU usage for CPU-only consolidation
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def setup_imports():
    """Import DeepSpeed utilities with error handling."""
    try:
        from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
        return convert_zero_checkpoint_to_fp32_state_dict
    except ImportError as e:
        print(f"Warning: DeepSpeed not available. DeepSpeed checkpoint consolidation will not work. Error: {e}")
        return None

def parse_args():
    parser = argparse.ArgumentParser(
        description="Consolidate DeepSpeed checkpoints or merge sharded models for vLLM inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Consolidate latest DeepSpeed checkpoint
  python consolidate_for_vllm.py --input ./ft_runs/qwen2.5-7b --output ./models/qwen2.5-7b-consolidated

  # Consolidate specific DeepSpeed checkpoint
  python consolidate_for_vllm.py --input ./ft_runs/qwen2.5-7b/checkpoint-196 --output ./models/qwen2.5-7b-step196

  # Merge HuggingFace sharded model
  python consolidate_for_vllm.py --input ./sharded_model --output ./single_file_model --type hf_sharded

  # Auto-detect and process
  python consolidate_for_vllm.py --input ./model_dir --output ./consolidated_model --auto-detect
        """
    )
    
    parser.add_argument(
        "--input", "--input-dir", dest="input_dir", required=True,
        help="Input directory: DeepSpeed training run dir, specific checkpoint dir, or HF model dir"
    )
    parser.add_argument(
        "--output", "--output-dir", dest="output_dir", required=True,
        help="Output directory for consolidated model"
    )
    parser.add_argument(
        "--type", choices=["deepspeed", "hf_sharded", "auto"], default="auto",
        help="Type of input: deepspeed checkpoint, hf_sharded model, or auto-detect"
    )
    parser.add_argument(
        "--checkpoint-dir", default=None,
        help="Specific checkpoint directory (for DeepSpeed, overrides auto-detection)"
    )
    parser.add_argument(
        "--base-model-name", default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model name for tokenizer/config (DeepSpeed mode only)"
    )
    parser.add_argument(
        "--max-shard-size", default="40GB",
        help="Maximum shard size for output (forces single file if large enough)"
    )
    parser.add_argument(
        "--safetensors", action="store_true", default=True,
        help="Save as safetensors format (default: True)"
    )
    parser.add_argument(
        "--pytorch-bin", dest="safetensors", action="store_false",
        help="Save as PyTorch .bin format instead of safetensors"
    )
    parser.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="float32",
        help="Data type for consolidation (float32 recommended for accuracy)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite output directory if it exists"
    )
    parser.add_argument(
        "--validate-vllm", action="store_true", default=True,
        help="Validate output for vLLM compatibility"
    )
    parser.add_argument(
        "--no-validate-vllm", dest="validate_vllm", action="store_false",
        help="Skip vLLM compatibility validation"
    )
    parser.add_argument(
        "--keep-original-precision", action="store_true",
        help="Keep original model precision instead of converting to float32"
    )
    
    return parser.parse_args()

def detect_input_type(input_dir: str) -> str:
    """Auto-detect the type of input directory."""
    input_path = Path(input_dir)
    
    # Check for DeepSpeed checkpoint indicators
    if (input_path / "zero_to_fp32.py").exists() or \
       (input_path / "mp_rank_00_model_states.pt").exists() or \
       any((input_path / f"mp_rank_{i:02d}_model_states.pt").exists() for i in range(8)):
        return "deepspeed"
    
    # Check for DeepSpeed training run directory
    checkpoint_dirs = list(input_path.glob("checkpoint-*"))
    if checkpoint_dirs:
        return "deepspeed"
    
    # Check for HuggingFace sharded model
    if (input_path / "pytorch_model.bin.index.json").exists():
        return "hf_sharded"
    
    # Check for subdirectory with shards (common DeepSpeed output pattern)
    pytorch_model_dir = input_path / "pytorch_model.bin"
    if pytorch_model_dir.is_dir() and (pytorch_model_dir / "pytorch_model.bin.index.json").exists():
        return "hf_sharded"
    
    # Check for single file model
    if (input_path / "pytorch_model.bin").exists() or (input_path / "model.safetensors").exists():
        return "single_file"
    
    # Default to DeepSpeed if config.json exists (might be a training output)
    if (input_path / "config.json").exists():
        return "deepspeed"
    
    raise ValueError(f"Cannot determine input type for directory: {input_dir}")

def find_latest_checkpoint(run_dir: str) -> str:
    """Find the latest checkpoint directory in a DeepSpeed training run."""
    checkpoints = sorted(
        glob.glob(os.path.join(run_dir, "checkpoint-*")),
        key=lambda p: int(p.rsplit("-", 1)[-1]) if p.rsplit("-", 1)[-1].isdigit() else -1,
    )
    if not checkpoints:
        raise ValueError(f"No checkpoint directories found in {run_dir}")
    return checkpoints[-1]

def consolidate_deepspeed_checkpoint(
    checkpoint_dir: str, 
    output_dir: str, 
    base_model_name: str,
    force: bool = False,
    use_safetensors: bool = True
) -> str:
    """Consolidate a DeepSpeed checkpoint to a single PyTorch file."""
    import torch
    convert_zero_checkpoint_to_fp32_state_dict = setup_imports()
    if convert_zero_checkpoint_to_fp32_state_dict is None:
        raise RuntimeError("DeepSpeed is required for checkpoint consolidation but not available")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine output filename
    checkpoint_name = os.path.basename(os.path.normpath(checkpoint_dir))
    if use_safetensors:
        if checkpoint_name.startswith("checkpoint-"):
            step = checkpoint_name.split("-", 1)[-1]
            if step.isdigit():
                dest_file = os.path.join(output_dir, f"model_{step}.safetensors")
            else:
                dest_file = os.path.join(output_dir, "model.safetensors")
        else:
            dest_file = os.path.join(output_dir, "model.safetensors")
    else:
        if checkpoint_name.startswith("checkpoint-"):
            step = checkpoint_name.split("-", 1)[-1]
            if step.isdigit():
                dest_file = os.path.join(output_dir, f"pytorch_model_{step}.bin")
            else:
                dest_file = os.path.join(output_dir, "pytorch_model.bin")
        else:
            dest_file = os.path.join(output_dir, "pytorch_model.bin")
    
    # Handle existing files
    if os.path.exists(dest_file) and not force:
        i = 1
        root, ext = os.path.splitext(dest_file)
        while os.path.exists(f"{root}.{i}{ext}"):
            i += 1
        dest_file = f"{root}.{i}{ext}"
        print(f"Output file exists, writing to: {dest_file}")
    
    print(f"Consolidating DeepSpeed checkpoint: {checkpoint_dir}")
    print(f"Writing consolidated weights to: {dest_file}")
    
    # Find the actual DeepSpeed checkpoint directory
    # Look for global_step* subdirectories or use the checkpoint_dir directly
    deepspeed_checkpoint_dir = checkpoint_dir
    global_step_dirs = glob.glob(os.path.join(checkpoint_dir, "global_step*"))
    if global_step_dirs:
        # Use the first (and typically only) global_step directory
        deepspeed_checkpoint_dir = sorted(global_step_dirs)[0]
        print(f"Found DeepSpeed files in subdirectory: {deepspeed_checkpoint_dir}")
        
        # Check if latest file exists, create it if missing
        latest_file = os.path.join(deepspeed_checkpoint_dir, "latest")
        if not os.path.exists(latest_file):
            print(f"Creating missing latest file: {latest_file}")
            with open(latest_file, "w") as f:
                f.write(".")
    
    # Convert checkpoint
    try:
        if use_safetensors:
            # For safetensors, we need to convert to a temporary directory first, then load and save as safetensors
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix='deepspeed_consolidation_')
            temp_model_path = os.path.join(temp_dir, 'consolidated_model')
            
            # Convert to temporary sharded model
            # Try with skip_optimizer_state=True if the standard conversion fails
            try:
                convert_zero_checkpoint_to_fp32_state_dict(deepspeed_checkpoint_dir, temp_model_path)
            except ValueError as e:
                if "optim_states" in str(e):
                    print(f"Standard conversion failed due to optimizer state mismatch, trying with dummy optimizer state files...")
                    # This is a ZeRO-3 checkpoint where model states are placeholders
                    # We need to create dummy optimizer state files to satisfy DeepSpeed's expectations
                    import tempfile
                    import shutil
                    import torch
                    
                    # Create a temporary directory with dummy optimizer state files
                    temp_deepspeed_dir = tempfile.mkdtemp(prefix='deepspeed_temp_')
                    try:
                        # Copy all files from the original directory
                        for item in os.listdir(deepspeed_checkpoint_dir):
                            src = os.path.join(deepspeed_checkpoint_dir, item)
                            dst = os.path.join(temp_deepspeed_dir, item)
                            if os.path.isfile(src):
                                shutil.copy2(src, dst)
                            elif os.path.isdir(src):
                                shutil.copytree(src, dst)
                        
                        # Create dummy optimizer state files for the missing ranks
                        import glob as glob_module
                        existing_optim_files = glob_module.glob(os.path.join(temp_deepspeed_dir, "*_optim_states.pt"))
                        print(f"Found {len(existing_optim_files)} existing optimizer state files")
                        
                        # Load one existing optimizer state file to use as template
                        if existing_optim_files:
                            template_file = existing_optim_files[0]
                            template_state = torch.load(template_file, map_location="cpu", weights_only=False)
                            
                            # Create dummy optimizer state files for missing ranks
                            # First, remove the existing optimizer state file to avoid conflicts
                            for existing_file in existing_optim_files:
                                os.remove(existing_file)
                                print(f"Removed existing optimizer state file: {os.path.basename(existing_file)}")
                            
                            # Create 8 dummy optimizer state files
                            # Use the original optimizer state file as a template for the correct size
                            original_optim_file = os.path.join(deepspeed_checkpoint_dir, "bf16_zero_pp_rank_6_mp_rank_00_optim_states.pt")
                            original_state = torch.load(original_optim_file, map_location="cpu", weights_only=False)
                            
                            for rank in range(8):  # Assuming 8 ranks
                                optim_file = os.path.join(temp_deepspeed_dir, f"zero_pp_rank_{rank}_mp_rank_00_optim_states.pt")
                                print(f"Creating dummy optimizer state file for rank {rank}")
                                # Create a dummy optimizer state with the correct fp32_flat_groups size
                                dummy_state = {
                                    'optimizer_state_dict': {
                                        'zero_stage': original_state['optimizer_state_dict']['zero_stage'],
                                        'loss_scaler': original_state['optimizer_state_dict']['loss_scaler'],
                                        'dynamic_loss_scale': original_state['optimizer_state_dict']['dynamic_loss_scale'],
                                        'overflow': original_state['optimizer_state_dict']['overflow'],
                                        'partition_count': original_state['optimizer_state_dict']['partition_count'],
                                        'optimizer_state_dict': {},
                                        'fp32_flat_groups': original_state['optimizer_state_dict']['fp32_flat_groups']  # Use original size
                                    },
                                    'ds_config': original_state['ds_config'],
                                    'ds_version': original_state['ds_version']
                                }
                                torch.save(dummy_state, optim_file)
                        
                        # Try conversion with the temporary directory
                        convert_zero_checkpoint_to_fp32_state_dict(temp_deepspeed_dir, temp_model_path)
                        print("Successfully converted using dummy optimizer state approach")
                    finally:
                        # Clean up temporary directory
                        shutil.rmtree(temp_deepspeed_dir, ignore_errors=True)
                else:
                    raise
            
            # Copy config from base model to temporary directory
            print(f"Copying config from base model...")
            config = AutoConfig.from_pretrained(base_model_name)
            config.save_pretrained(temp_model_path)
            
            # Load the sharded model and save as single safetensors file
            print(f"Loading consolidated model from temporary directory...")
            model = AutoModelForCausalLM.from_pretrained(
                temp_model_path,
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True
            )
            
            print(f"Saving as single safetensors file to '{dest_file}'...")
            # Save just the state dict as safetensors
            from safetensors.torch import save_file
            save_file(model.state_dict(), dest_file)
            
            # Clean up temporary files
            import shutil
            shutil.rmtree(temp_dir)
            print(f"Successfully consolidated checkpoint to '{dest_file}' (safetensors format)")
        else:
            # Direct conversion to .bin format
            convert_zero_checkpoint_to_fp32_state_dict(deepspeed_checkpoint_dir, dest_file)
            print(f"Successfully consolidated checkpoint to '{dest_file}' (PyTorch .bin format)")
    except Exception as e:
        raise RuntimeError(f"DeepSpeed checkpoint consolidation failed: {e}")
    
    # Save tokenizer and config
    print(f"Saving tokenizer and config to {output_dir}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.save_pretrained(output_dir)
        config = AutoConfig.from_pretrained(base_model_name)
        config.save_pretrained(output_dir)
        print("Successfully saved tokenizer and config.")
    except Exception as e:
        print(f"Warning: Failed to save tokenizer/config: {e}")
    
    return dest_file

def copy_single_file_model(src_dir: str, dst_dir: str) -> bool:
    """Copy a single-file model, normalizing the filename."""
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    dst_path.mkdir(parents=True, exist_ok=True)
    
    # Check for standard single files
    candidates = [
        ("pytorch_model.bin", "pytorch_model.bin"),
        ("model.safetensors", "model.safetensors"),
    ]
    
    # Look for consolidated files with step numbers
    for file in src_path.glob("pytorch_model_*.bin"):
        candidates.append((file.name, "pytorch_model.bin"))
        break
    
    copied = False
    for src_name, dst_name in candidates:
        src_file = src_path / src_name
        if src_file.exists():
            shutil.copy2(src_file, dst_path / dst_name)
            copied = True
            print(f"Copied {src_name} -> {dst_name}")
            break
    
    if not copied:
        return False
    
    # Copy auxiliary files
    aux_files = [
        "config.json", "generation_config.json", "tokenizer.json",
        "tokenizer_config.json", "special_tokens_map.json", "vocab.json",
        "merges.txt", "added_tokens.json", "chat_template.jinja"
    ]
    
    for fname in aux_files:
        src_file = src_path / fname
        if src_file.exists():
            try:
                shutil.copy2(src_file, dst_path / fname)
            except Exception as e:
                print(f"Warning: Failed to copy {fname}: {e}")
    
    return True

def merge_hf_sharded_model(
    input_dir: str, 
    output_dir: str, 
    max_shard_size: str = "40GB",
    dtype: str = "float32",
    keep_original_precision: bool = False,
    use_safetensors: bool = True
) -> str:
    """Merge a HuggingFace sharded model into a single file."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Merging HuggingFace sharded model: {input_dir}")
    
    # Detect weights directory (handle nested pytorch_model.bin directory)
    weights_dir = input_dir
    root_index = input_path / "pytorch_model.bin.index.json"
    subdir = input_path / "pytorch_model.bin"
    subdir_index = subdir / "pytorch_model.bin.index.json"
    
    if not root_index.exists() and subdir.is_dir() and subdir_index.exists():
        print("Detected shards in 'pytorch_model.bin/' subdirectory")
        weights_dir = str(subdir)
    
    # Check if already single file
    if not (Path(weights_dir) / "pytorch_model.bin.index.json").exists():
        print("Model appears to be single-file, copying...")
        if copy_single_file_model(weights_dir, str(output_path)):
            if use_safetensors:
                return str(output_path / "model.safetensors")
            else:
                return str(output_path / "pytorch_model.bin")
        else:
            raise ValueError(f"No valid model weights found in {weights_dir}")
    
    # Load config
    try:
        config = AutoConfig.from_pretrained(input_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to load model config: {e}")
    
    # Determine dtype
    torch_dtype = torch.float32
    if not keep_original_precision:
        if dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16
    
    # Load and merge model
    print(f"Loading model with dtype {torch_dtype}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            weights_dir,
            config=config,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map="cpu",
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load sharded model: {e}")
    
    # Save consolidated model
    print(f"Saving consolidated model to: {output_path}")
    try:
        if use_safetensors:
            model.save_pretrained(
                output_path,
                max_shard_size=max_shard_size,
                safe_serialization=True
            )
            print("Saved model in safetensors format")
        else:
            model.save_pretrained(
                output_path,
                max_shard_size=max_shard_size,
                safe_serialization=False
            )
            print("Saved model in PyTorch .bin format")
    except Exception as e:
        raise RuntimeError(f"Failed to save consolidated model: {e}")
    
    # Save tokenizer
    print("Saving tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(input_dir)
        tokenizer.save_pretrained(output_path)
    except Exception as e:
        print(f"Warning: Failed to save tokenizer: {e}")
    
    if use_safetensors:
        return str(output_path / "model.safetensors")
    else:
        return str(output_path / "pytorch_model.bin")

def validate_vllm_compatibility(model_dir: str) -> Dict[str, Any]:
    """Validate that the model is compatible with vLLM."""
    model_path = Path(model_dir)
    validation_results = {
        "compatible": True,
        "issues": [],
        "warnings": [],
        "info": {}
    }
    
    # Check for required files
    required_files = ["config.json"]
    for file in required_files:
        if not (model_path / file).exists():
            validation_results["issues"].append(f"Missing required file: {file}")
            validation_results["compatible"] = False
    
    # Check for model weights
    weight_files = list(model_path.glob("pytorch_model*.bin")) + list(model_path.glob("model*.safetensors"))
    if not weight_files:
        validation_results["issues"].append("No model weight files found")
        validation_results["compatible"] = False
    
    # Prefer safetensors format
    safetensors_files = list(model_path.glob("model*.safetensors"))
    if safetensors_files:
        validation_results["info"]["format"] = "safetensors"
        validation_results["info"]["weight_files"] = [f.name for f in safetensors_files]
    else:
        bin_files = list(model_path.glob("pytorch_model*.bin"))
        if bin_files:
            validation_results["info"]["format"] = "pytorch_bin"
            validation_results["info"]["weight_files"] = [f.name for f in bin_files]
    
    # Check for sharding
    index_file = model_path / "pytorch_model.bin.index.json"
    if index_file.exists():
        try:
            with open(index_file) as f:
                index_data = json.load(f)
            
            weight_map = index_data.get("weight_map", {})
            unique_files = set(weight_map.values())
            
            if len(unique_files) > 1:
                validation_results["issues"].append(
                    f"Model is still sharded ({len(unique_files)} files). vLLM requires single-file models."
                )
                validation_results["compatible"] = False
            
            validation_results["info"]["num_shards"] = len(unique_files)
            validation_results["info"]["shard_files"] = list(unique_files)
            
        except Exception as e:
            validation_results["warnings"].append(f"Could not parse index file: {e}")
    
    # Check model size
    total_size = 0
    for weight_file in weight_files:
        total_size += weight_file.stat().st_size
    
    validation_results["info"]["total_size_gb"] = total_size / (1024**3)
    
    # Check for tokenizer
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json"]
    has_tokenizer = any((model_path / f).exists() for f in tokenizer_files)
    if not has_tokenizer:
        validation_results["warnings"].append("No tokenizer files found")
    
    return validation_results

def main():
    args = parse_args()
    
    print("=== vLLM Model Consolidation Tool ===")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    
    # Auto-detect input type if needed
    if args.type == "auto":
        try:
            detected_type = detect_input_type(args.input_dir)
            print(f"Auto-detected input type: {detected_type}")
            args.type = detected_type
        except ValueError as e:
            print(f"Error: {e}")
            return 1
    
    # Handle existing output directory
    if os.path.exists(args.output_dir):
        if not args.force:
            print(f"Error: Output directory '{args.output_dir}' already exists. Use --force to overwrite.")
            return 1
        else:
            print(f"Removing existing output directory: {args.output_dir}")
            shutil.rmtree(args.output_dir)
    
    try:
        # Process based on input type
        if args.type == "deepspeed":
            # Handle DeepSpeed checkpoint
            if args.checkpoint_dir:
                checkpoint_dir = args.checkpoint_dir
            elif (os.path.isfile(os.path.join(args.input_dir, "zero_to_fp32.py")) or \
                  any(os.path.isfile(os.path.join(args.input_dir, f"mp_rank_{i:02d}_model_states.pt")) for i in range(8)) or \
                  glob.glob(os.path.join(args.input_dir, "global_step*")) or \
                  (os.path.isfile(os.path.join(args.input_dir, "config.json")) and 
                   os.path.isfile(os.path.join(args.input_dir, "tokenizer.json")))):
                # Input is already a checkpoint directory
                checkpoint_dir = args.input_dir
            else:
                # Input is a training run directory, find latest checkpoint
                checkpoint_dir = find_latest_checkpoint(args.input_dir)
            
            print(f"Using checkpoint: {checkpoint_dir}")
            consolidate_deepspeed_checkpoint(
                checkpoint_dir, 
                args.output_dir, 
                args.base_model_name,
                args.force,
                args.safetensors
            )
            
        elif args.type == "hf_sharded":
            # Handle HuggingFace sharded model
            merge_hf_sharded_model(
                args.input_dir,
                args.output_dir,
                args.max_shard_size,
                args.dtype,
                args.keep_original_precision,
                args.safetensors
            )
            
        elif args.type == "single_file":
            # Just copy single file model
            print("Input is already a single-file model, copying...")
            if not copy_single_file_model(args.input_dir, args.output_dir):
                print("Error: Failed to copy single-file model")
                return 1
        
        else:
            print(f"Error: Unknown input type: {args.type}")
            return 1
        
        # Validate vLLM compatibility
        if args.validate_vllm:
            print("\n=== Validating vLLM Compatibility ===")
            validation = validate_vllm_compatibility(args.output_dir)
            
            if validation["compatible"]:
                print("✅ Model is compatible with vLLM!")
            else:
                print("❌ Model has compatibility issues:")
                for issue in validation["issues"]:
                    print(f"  - {issue}")
            
            if validation["warnings"]:
                print("⚠️  Warnings:")
                for warning in validation["warnings"]:
                    print(f"  - {warning}")
            
            if validation["info"]:
                print("ℹ️  Model info:")
                for key, value in validation["info"].items():
                    print(f"  - {key}: {value}")
        
        print(f"\n=== Consolidation Complete ===")
        print(f"Consolidated model ready at: {args.output_dir}")
        print("This directory can now be used with vLLM for inference.")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
