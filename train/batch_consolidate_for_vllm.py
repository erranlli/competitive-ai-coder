#!/usr/bin/env python3
"""
Batch processing script for consolidating multiple DeepSpeed checkpoints or models for vLLM inference.

This script can process:
1. Multiple training runs in parallel
2. All checkpoints from a single training run
3. Multiple model directories
4. Custom checkpoint selection patterns

Useful for processing large numbers of models efficiently.
"""

import os
import sys
import glob
import argparse
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch consolidate multiple models/checkpoints for vLLM inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all training runs in a directory
  python batch_consolidate_for_vllm.py --input-pattern "./ft_runs_*/qwen2.5-7b" --output-dir ./consolidated_models

  # Process all checkpoints from a single run
  python batch_consolidate_for_vllm.py --input-dir ./ft_runs/qwen2.5-7b --output-dir ./all_checkpoints --all-checkpoints

  # Process specific checkpoint patterns
  python batch_consolidate_for_vllm.py --input-pattern "./ft_runs/*/checkpoint-{100,200,300}" --output-dir ./selected_checkpoints

  # Process with custom naming
  python batch_consolidate_for_vllm.py --input-dir ./ft_runs --output-dir ./models --naming-template "{run_name}_step_{step}"
        """
    )
    
    # Input specification
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-dir", 
        help="Single input directory (training run or model directory)"
    )
    input_group.add_argument(
        "--input-pattern", 
        help="Glob pattern for multiple input directories"
    )
    input_group.add_argument(
        "--input-list", 
        help="File containing list of input directories (one per line)"
    )
    
    # Output specification
    parser.add_argument(
        "--output-dir", required=True,
        help="Base output directory for consolidated models"
    )
    
    # Processing options
    parser.add_argument(
        "--all-checkpoints", action="store_true",
        help="Process all checkpoints in each training run (not just the latest)"
    )
    parser.add_argument(
        "--checkpoint-filter",
        help="Filter checkpoints by pattern (e.g., 'checkpoint-{100,200,300}' or 'checkpoint-*[05]0')"
    )
    parser.add_argument(
        "--max-workers", type=int, default=4,
        help="Maximum number of parallel workers"
    )
    parser.add_argument(
        "--naming-template", default="{input_name}_step_{step}",
        help="Template for output directory names. Available: {input_name}, {run_name}, {step}, {checkpoint_name}"
    )
    
    # Model options
    parser.add_argument(
        "--base-model-name", default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model name for tokenizer/config (DeepSpeed mode only)"
    )
    parser.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="float32",
        help="Data type for consolidation"
    )
    parser.add_argument(
        "--max-shard-size", default="40GB",
        help="Maximum shard size for output"
    )
    parser.add_argument(
        "--safetensors", action="store_true", default=True,
        help="Save as safetensors format (default: True)"
    )
    parser.add_argument(
        "--pytorch-bin", dest="safetensors", action="store_false",
        help="Save as PyTorch .bin format instead of safetensors"
    )
    
    # Control options
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing output directories"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be processed without actually doing it"
    )
    parser.add_argument(
        "--continue-on-error", action="store_true",
        help="Continue processing other models if one fails"
    )
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Skip vLLM compatibility validation"
    )
    
    return parser.parse_args()

def find_inputs(args) -> List[str]:
    """Find all input directories to process."""
    inputs = []
    
    if args.input_dir:
        inputs.append(args.input_dir)
    elif args.input_pattern:
        inputs.extend(glob.glob(args.input_pattern))
    elif args.input_list:
        with open(args.input_list) as f:
            inputs.extend(line.strip() for line in f if line.strip())
    
    # Validate inputs exist
    valid_inputs = []
    for input_dir in inputs:
        if os.path.isdir(input_dir):
            valid_inputs.append(input_dir)
        else:
            print(f"Warning: Input directory does not exist: {input_dir}")
    
    return valid_inputs

def find_checkpoints(input_dir: str, all_checkpoints: bool = False, checkpoint_filter: Optional[str] = None) -> List[str]:
    """Find checkpoints to process in an input directory."""
    input_path = Path(input_dir)
    checkpoints = []
    
    # Check if input_dir is itself a checkpoint
    if (input_path / "zero_to_fp32.py").exists() or \
       any((input_path / f"mp_rank_{i:02d}_model_states.pt").exists() for i in range(8)):
        return [str(input_path)]
    
    # Find checkpoint directories
    if checkpoint_filter:
        pattern = checkpoint_filter
    else:
        pattern = "checkpoint-*"
    
    checkpoint_dirs = list(input_path.glob(pattern))
    
    if not checkpoint_dirs:
        # Maybe it's a regular model directory
        return [str(input_path)]
    
    # Sort by checkpoint number
    def checkpoint_key(path):
        name = path.name
        if name.startswith("checkpoint-"):
            try:
                return int(name.split("-", 1)[1])
            except ValueError:
                return 0
        return 0
    
    checkpoint_dirs.sort(key=checkpoint_key)
    
    if all_checkpoints:
        checkpoints = [str(p) for p in checkpoint_dirs]
    else:
        # Just the latest
        if checkpoint_dirs:
            checkpoints = [str(checkpoint_dirs[-1])]
    
    return checkpoints

def generate_output_name(input_dir: str, checkpoint_dir: str, template: str) -> str:
    """Generate output directory name from template."""
    input_path = Path(input_dir)
    checkpoint_path = Path(checkpoint_dir)
    
    # Extract components
    input_name = input_path.name
    run_name = input_path.name
    checkpoint_name = checkpoint_path.name
    
    # Extract step number
    step = "unknown"
    if checkpoint_name.startswith("checkpoint-"):
        step_part = checkpoint_name.split("-", 1)[1]
        if step_part.isdigit():
            step = step_part
    
    # If checkpoint is same as input, it's not a nested checkpoint
    if checkpoint_path == input_path:
        checkpoint_name = "model"
        if step == "unknown":
            step = "final"
    
    # Apply template
    name = template.format(
        input_name=input_name,
        run_name=run_name,
        step=step,
        checkpoint_name=checkpoint_name
    )
    
    # Sanitize filename
    name = "".join(c for c in name if c.isalnum() or c in "._-")
    return name

def consolidate_single_model(
    input_dir: str,
    checkpoint_dir: str,
    output_dir: str,
    args
) -> Dict[str, Any]:
    """Consolidate a single model/checkpoint."""
    result = {
        "input_dir": input_dir,
        "checkpoint_dir": checkpoint_dir,
        "output_dir": output_dir,
        "success": False,
        "error": None,
        "start_time": time.time(),
        "end_time": None,
        "duration": None
    }
    
    try:
        # Build command - use rllm environment python if available
        python_executable = sys.executable
        rllm_python = "/home/g10/anaconda3/envs/rllm/bin/python"
        if os.path.exists(rllm_python):
            python_executable = rllm_python
        
        cmd = [
            python_executable,
            os.path.join(os.path.dirname(__file__), "consolidate_for_vllm.py"),
            "--input", checkpoint_dir,
            "--output", output_dir,
            "--base-model-name", args.base_model_name,
            "--dtype", args.dtype,
            "--max-shard-size", args.max_shard_size,
        ]
        
        if not args.safetensors:
            cmd.append("--pytorch-bin")
        
        if args.force:
            cmd.append("--force")
        
        if args.skip_validation:
            cmd.append("--no-validate-vllm")
        
        # Run consolidation
        print(f"Processing: {checkpoint_dir} -> {output_dir}")
        
        if args.dry_run:
            print(f"[DRY RUN] Would run: {' '.join(cmd)}")
            result["success"] = True
        else:
            process_result = subprocess.run(cmd, capture_output=True, text=True)
            if process_result.returncode == 0:
                result["success"] = True
                print(f"✅ Completed: {output_dir}")
            else:
                result["success"] = False
                stderr_msg = process_result.stderr.strip() if process_result.stderr else "No error message"
                stdout_msg = process_result.stdout.strip() if process_result.stdout else "No output"
                result["error"] = f"Command failed (exit code {process_result.returncode}): {stderr_msg}"
                result["stdout"] = stdout_msg
                print(f"❌ Failed: {checkpoint_dir}")
                print(f"   Error: {stderr_msg}")
                if stdout_msg and stdout_msg != "No output":
                    print(f"   Output: {stdout_msg}")
        
    except Exception as e:
        result["error"] = str(e)
        print(f"❌ Failed: {checkpoint_dir} - {result['error']}")
    
    result["end_time"] = time.time()
    result["duration"] = result["end_time"] - result["start_time"]
    
    return result

def main():
    args = parse_args()
    
    print("=== Batch vLLM Model Consolidation ===")
    
    # Find all inputs
    input_dirs = find_inputs(args)
    if not input_dirs:
        print("Error: No valid input directories found")
        return 1
    
    print(f"Found {len(input_dirs)} input directories")
    
    # Build task list
    tasks = []
    for input_dir in input_dirs:
        checkpoints = find_checkpoints(
            input_dir, 
            args.all_checkpoints, 
            args.checkpoint_filter
        )
        
        for checkpoint_dir in checkpoints:
            output_name = generate_output_name(input_dir, checkpoint_dir, args.naming_template)
            output_path = os.path.join(args.output_dir, output_name)
            
            tasks.append({
                "input_dir": input_dir,
                "checkpoint_dir": checkpoint_dir,
                "output_dir": output_path
            })
    
    print(f"Found {len(tasks)} models/checkpoints to process")
    
    if args.dry_run:
        print("\n=== DRY RUN - Tasks to be processed ===")
        for i, task in enumerate(tasks, 1):
            print(f"{i:3d}. {task['checkpoint_dir']} -> {task['output_dir']}")
        return 0
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process tasks
    results = []
    failed_count = 0
    
    if args.max_workers == 1:
        # Sequential processing
        for task in tasks:
            result = consolidate_single_model(
                task["input_dir"],
                task["checkpoint_dir"],
                task["output_dir"],
                args
            )
            results.append(result)
            
            if not result["success"]:
                failed_count += 1
                if not args.continue_on_error:
                    print("Stopping due to error (use --continue-on-error to continue)")
                    break
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(
                    consolidate_single_model,
                    task["input_dir"],
                    task["checkpoint_dir"],
                    task["output_dir"],
                    args
                ): task for task in tasks
            }
            
            # Collect results
            for future in as_completed(future_to_task):
                result = future.result()
                results.append(result)
                
                if not result["success"]:
                    failed_count += 1
                    if not args.continue_on_error:
                        # Cancel remaining tasks
                        for f in future_to_task:
                            f.cancel()
                        break
    
    # Generate summary
    print(f"\n=== Processing Summary ===")
    print(f"Total tasks: {len(tasks)}")
    print(f"Completed: {len(results) - failed_count}")
    print(f"Failed: {failed_count}")
    
    if results:
        total_time = sum(r["duration"] for r in results if r["duration"])
        avg_time = total_time / len(results) if results else 0
        print(f"Total time: {total_time:.1f}s")
        print(f"Average time per model: {avg_time:.1f}s")
    
    # Save detailed results
    results_file = os.path.join(args.output_dir, "batch_consolidation_results.json")
    with open(results_file, "w") as f:
        json.dump({
            "args": vars(args),
            "summary": {
                "total_tasks": len(tasks),
                "completed": len(results) - failed_count,
                "failed": failed_count,
                "total_time": total_time if results else 0,
                "average_time": avg_time if results else 0
            },
            "results": results
        }, f, indent=2)
    
    print(f"Detailed results saved to: {results_file}")
    
    if failed_count > 0:
        print(f"\n❌ {failed_count} models failed to process")
        return 1
    else:
        print(f"\n✅ All models processed successfully!")
        return 0

if __name__ == "__main__":
    exit(main())
