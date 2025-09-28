#!/usr/bin/env python3
"""
Batch inference script for multiple checkpoints using vLLM.

This script processes multiple consolidated checkpoints in parallel, using 4 GPUs per checkpoint
to maximize GPU utilization (2 checkpoints running simultaneously on 8 GPUs total).
"""

import os
import sys
import glob
import argparse
import subprocess
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch inference for multiple checkpoints using vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all consolidated checkpoints
  python infer/batch_inference_vllm.py \\
    --checkpoints-dir /mnt/data3/all_checkpoints_consolidated \\
    --output-dir /mnt/data3/model_solutions_consolidated \\
    --max-problems 64

  # Process specific checkpoints
  python infer/batch_inference_vllm.py \\
    --checkpoints-pattern "/mnt/data3/checkpoints/qwen2.5-7b_step_{196,392,588}" \\
    --output-dir /mnt/data3/model_solutions \\
    --max-problems 64

  # Custom GPU allocation
  python infer/batch_inference_vllm.py \\
    --checkpoints-dir /mnt/data3/checkpoints \\
    --output-dir /mnt/data3/results \\
    --gpus-per-checkpoint 4 \\
    --max-concurrent 2
        """
    )
    
    # Input specification
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--checkpoints-dir",
        help="Directory containing consolidated checkpoint directories"
    )
    input_group.add_argument(
        "--checkpoints-pattern",
        help="Glob pattern for checkpoint directories"
    )
    input_group.add_argument(
        "--checkpoints-list",
        help="File containing list of checkpoint directories (one per line)"
    )
    
    # Output and processing options
    parser.add_argument(
        "--output-dir", required=True,
        help="Base output directory for inference results"
    )
    parser.add_argument(
        "--max-problems", type=int, default=64,
        help="Maximum number of problems to process per checkpoint"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for inference"
    )
    
    # GPU allocation
    parser.add_argument(
        "--gpus-per-checkpoint", type=int, default=4,
        help="Number of GPUs to use per checkpoint (default: 4)"
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=2,
        help="Maximum number of concurrent checkpoint inferences (default: 2)"
    )
    parser.add_argument(
        "--gpu-ids", default="0,1,2,3,4,5,6,7",
        help="Available GPU IDs (default: 0,1,2,3,4,5,6,7)"
    )
    
    # Model and inference options
    parser.add_argument(
        "--base-model-name", default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model name for reference"
    )
    parser.add_argument(
        "--dataset-name", default="open-r1/codeforces",
        help="Dataset name for inference"
    )
    parser.add_argument(
        "--dataset-subset", default="default",
        help="Dataset subset"
    )
    parser.add_argument(
        "--dataset-split", default="test",
        help="Dataset split"
    )
    parser.add_argument(
        "--max-model-len", type=int, default=32768,
        help="Maximum model sequence length"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=32000,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--dtype", default="bfloat16",
        help="Model dtype for inference"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p", type=float, default=0.95,
        help="Top-p sampling parameter"
    )
    
    # Control options
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be processed without running inference"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing results"
    )
    parser.add_argument(
        "--continue-on-error", action="store_true",
        help="Continue processing if individual checkpoints fail"
    )
    parser.add_argument(
        "--checkpoint-filter",
        help="Filter checkpoints by pattern (e.g., '*step_{196,392,588}*')"
    )
    parser.add_argument(
        "--sort-order", choices=["step", "name", "none"], default="step",
        help="Sort order for processing checkpoints (default: step)"
    )
    
    return parser.parse_args()

class GPUAllocator:
    """Manages GPU allocation for concurrent inference processes."""
    
    def __init__(self, available_gpus: List[int], gpus_per_job: int):
        self.available_gpus = available_gpus
        self.gpus_per_job = gpus_per_job
        self.lock = threading.Lock()
        self.allocated_gpus = set()
    
    def allocate_gpus(self) -> Optional[List[int]]:
        """Allocate GPUs for a job. Returns None if not enough GPUs available."""
        with self.lock:
            available = [gpu for gpu in self.available_gpus if gpu not in self.allocated_gpus]
            if len(available) >= self.gpus_per_job:
                allocated = available[:self.gpus_per_job]
                self.allocated_gpus.update(allocated)
                return allocated
            return None
    
    def release_gpus(self, gpus: List[int]):
        """Release GPUs back to the available pool."""
        with self.lock:
            self.allocated_gpus.difference_update(gpus)

def find_checkpoints(args) -> List[str]:
    """Find all checkpoint directories to process."""
    checkpoints = []
    
    if args.checkpoints_dir:
        # Find all subdirectories in checkpoints_dir
        pattern = os.path.join(args.checkpoints_dir, "*")
        candidates = glob.glob(pattern)
        checkpoints.extend([p for p in candidates if os.path.isdir(p)])
    elif args.checkpoints_pattern:
        candidates = glob.glob(args.checkpoints_pattern)
        checkpoints.extend([p for p in candidates if os.path.isdir(p)])
    elif args.checkpoints_list:
        with open(args.checkpoints_list) as f:
            checkpoints.extend([line.strip() for line in f if line.strip()])
    
    # Apply filter if specified
    if args.checkpoint_filter:
        import fnmatch
        checkpoints = [cp for cp in checkpoints if fnmatch.fnmatch(os.path.basename(cp), args.checkpoint_filter)]
    
    # Validate checkpoints have model files
    valid_checkpoints = []
    for cp in checkpoints:
        cp_path = Path(cp)
        has_model = any([
            (cp_path / "model.safetensors").exists(),
            (cp_path / "pytorch_model.bin").exists(),
            list(cp_path.glob("pytorch_model-*.bin")),
            list(cp_path.glob("model_*.safetensors"))
        ])
        if has_model:
            valid_checkpoints.append(cp)
        else:
            print(f"Warning: No model files found in {cp}")
    
    # Sort by step number for proper ordering
    def checkpoint_sort_key(checkpoint_path):
        """Extract numeric step for proper sorting."""
        checkpoint_name = os.path.basename(checkpoint_path)
        
        # Try to extract step number
        if "step_" in checkpoint_name:
            try:
                step_str = checkpoint_name.split("step_")[1].split("_")[0].split("-")[0]
                return int(step_str)
            except (IndexError, ValueError):
                pass
        elif "checkpoint-" in checkpoint_name:
            try:
                step_str = checkpoint_name.split("checkpoint-")[1].split("_")[0].split("-")[0]
                return int(step_str)
            except (IndexError, ValueError):
                pass
        
        # Fallback to alphabetical sorting
        return float('inf'), checkpoint_name
    
    # Sort based on specified order
    if args.sort_order == "step":
        return sorted(valid_checkpoints, key=checkpoint_sort_key)
    elif args.sort_order == "name":
        return sorted(valid_checkpoints)
    else:  # "none"
        return valid_checkpoints

def extract_checkpoint_info(checkpoint_path: str) -> Dict[str, str]:
    """Extract checkpoint information from path."""
    checkpoint_name = os.path.basename(checkpoint_path)
    
    # Try to extract step number
    step = "unknown"
    if "step_" in checkpoint_name:
        try:
            step = checkpoint_name.split("step_")[1].split("_")[0].split("-")[0]
        except (IndexError, ValueError):
            pass
    elif "checkpoint-" in checkpoint_name:
        try:
            step = checkpoint_name.split("checkpoint-")[1].split("_")[0].split("-")[0]
        except (IndexError, ValueError):
            pass
    
    return {
        "name": checkpoint_name,
        "step": step,
        "path": checkpoint_path
    }

def generate_output_dir(checkpoint_info: Dict[str, str], base_output_dir: str) -> str:
    """Generate output directory name for checkpoint results."""
    if checkpoint_info["step"] != "unknown":
        return os.path.join(base_output_dir, f"checkpoint-{checkpoint_info['step']}")
    else:
        return os.path.join(base_output_dir, checkpoint_info["name"])

def run_inference_for_checkpoint(
    checkpoint_path: str,
    output_dir: str,
    gpu_ids: List[int],
    args
) -> Dict[str, Any]:
    """Run inference for a single checkpoint."""
    result = {
        "checkpoint_path": checkpoint_path,
        "output_dir": output_dir,
        "gpu_ids": gpu_ids,
        "success": False,
        "error": None,
        "start_time": time.time(),
        "end_time": None,
        "duration": None
    }
    
    try:
        # Find the correct Python executable (same logic as consolidation)
        python_executable = sys.executable
        rllm_python = "/home/g10/anaconda3/envs/rllm/bin/python"
        if os.path.exists(rllm_python):
            python_executable = rllm_python
        
        # Build inference command
        gpu_ids_str = ",".join(map(str, gpu_ids))
        cmd = [
            python_executable,
            "infer/generate_qwen_vllm_think.py",
            "--dataset-name", args.dataset_name,
            "--subset", args.dataset_subset,
            "--split", args.dataset_split,
            "--model-name", args.base_model_name,
            "--checkpoint-path", checkpoint_path,
            "--batch-size", str(args.batch_size),
            "--max-model-len", str(args.max_model_len),
            "--max-new-tokens", str(args.max_new_tokens),
            "--tensor-parallel-size", str(len(gpu_ids)),
            "--gpu-ids", gpu_ids_str,
            "--dtype", args.dtype,
            "--temperature", str(args.temperature),
            "--top-p", str(args.top_p),
            "--results-dir", output_dir,
            "--max-problems", str(args.max_problems)
        ]
        
        print(f"🚀 Starting inference: {os.path.basename(checkpoint_path)} on GPUs {gpu_ids}")
        print(f"   Output: {output_dir}")
        
        if args.dry_run:
            print(f"[DRY RUN] Would run: {' '.join(cmd)}")
            result["success"] = True
        else:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Run inference
            process_result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            
            if process_result.returncode == 0:
                result["success"] = True
                print(f"✅ Completed: {os.path.basename(checkpoint_path)}")
            else:
                result["success"] = False
                stderr_msg = process_result.stderr.strip() if process_result.stderr else "No error message"
                result["error"] = f"Command failed (exit code {process_result.returncode}): {stderr_msg}"
                result["stdout"] = process_result.stdout.strip() if process_result.stdout else ""
                print(f"❌ Failed: {os.path.basename(checkpoint_path)}")
                print(f"   Error: {stderr_msg}")
                
    except Exception as e:
        result["error"] = str(e)
        print(f"❌ Failed: {os.path.basename(checkpoint_path)} - {result['error']}")
    
    result["end_time"] = time.time()
    result["duration"] = result["end_time"] - result["start_time"]
    
    return result

def main():
    args = parse_args()
    
    print("=== Batch vLLM Inference ===")
    
    # Parse GPU IDs
    available_gpus = [int(x.strip()) for x in args.gpu_ids.split(",")]
    print(f"Available GPUs: {available_gpus}")
    print(f"GPUs per checkpoint: {args.gpus_per_checkpoint}")
    print(f"Max concurrent checkpoints: {args.max_concurrent}")
    
    # Validate GPU allocation
    total_gpus_needed = args.max_concurrent * args.gpus_per_checkpoint
    if total_gpus_needed > len(available_gpus):
        print(f"Warning: Need {total_gpus_needed} GPUs but only {len(available_gpus)} available")
        args.max_concurrent = len(available_gpus) // args.gpus_per_checkpoint
        print(f"Reducing max_concurrent to {args.max_concurrent}")
    
    # Find checkpoints
    checkpoints = find_checkpoints(args)
    if not checkpoints:
        print("Error: No valid checkpoints found")
        return 1
    
    print(f"Found {len(checkpoints)} checkpoints to process")
    
    # Show processing order
    if len(checkpoints) > 0:
        print(f"Processing order (sorted by {args.sort_order}):")
        for i, checkpoint in enumerate(checkpoints[:5], 1):  # Show first 5
            checkpoint_info = extract_checkpoint_info(checkpoint)
            print(f"  {i:2d}. {checkpoint_info['name']} (step {checkpoint_info['step']})")
        if len(checkpoints) > 5:
            print(f"  ... and {len(checkpoints) - 5} more")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare tasks
    tasks = []
    for checkpoint_path in checkpoints:
        checkpoint_info = extract_checkpoint_info(checkpoint_path)
        output_dir = generate_output_dir(checkpoint_info, args.output_dir)
        
        # Check if results already exist
        if not args.force and os.path.exists(output_dir):
            result_files = glob.glob(os.path.join(output_dir, "*.jsonl"))
            if result_files:
                print(f"⏭️  Skipping {checkpoint_info['name']} (results exist)")
                continue
        
        tasks.append({
            "checkpoint_path": checkpoint_path,
            "checkpoint_info": checkpoint_info,
            "output_dir": output_dir
        })
    
    if not tasks:
        print("No tasks to process (all results already exist, use --force to overwrite)")
        return 0
    
    print(f"Processing {len(tasks)} tasks")
    
    if args.dry_run:
        print("\n=== DRY RUN - Tasks to be processed ===")
        for i, task in enumerate(tasks, 1):
            print(f"{i:3d}. {task['checkpoint_info']['name']} -> {task['output_dir']}")
        return 0
    
    # Initialize GPU allocator
    gpu_allocator = GPUAllocator(available_gpus, args.gpus_per_checkpoint)
    
    # Process tasks with limited concurrency
    results = []
    failed_count = 0
    
    def process_task_with_gpu_allocation(task):
        """Process a single task with GPU allocation."""
        # Wait for GPU allocation
        while True:
            allocated_gpus = gpu_allocator.allocate_gpus()
            if allocated_gpus is not None:
                break
            time.sleep(1)  # Wait before trying again
        
        try:
            # Run inference
            result = run_inference_for_checkpoint(
                task["checkpoint_path"],
                task["output_dir"],
                allocated_gpus,
                args
            )
            return result
        finally:
            # Always release GPUs
            gpu_allocator.release_gpus(allocated_gpus)
    
    # Process tasks with threading for better GPU management
    from concurrent.futures import ThreadPoolExecutor
    import threading
    
    results_lock = threading.Lock()
    
    def process_task_with_gpu_allocation(task):
        """Process a single task with GPU allocation."""
        # Wait for GPU allocation
        while True:
            allocated_gpus = gpu_allocator.allocate_gpus()
            if allocated_gpus is not None:
                break
            time.sleep(1)  # Wait before trying again
        
        try:
            # Run inference
            result = run_inference_for_checkpoint(
                task["checkpoint_path"],
                task["output_dir"],
                allocated_gpus,
                args
            )
            
            # Thread-safe result collection
            with results_lock:
                results.append(result)
            
            return result
        finally:
            # Always release GPUs
            gpu_allocator.release_gpus(allocated_gpus)
    
    # Use ThreadPoolExecutor for concurrent processing
    with ThreadPoolExecutor(max_workers=args.max_concurrent) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(process_task_with_gpu_allocation, task): task 
            for task in tasks
        }
        
        # Collect results
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                
                if not result["success"]:
                    failed_count += 1
                    if not args.continue_on_error:
                        # Cancel remaining tasks
                        for f in future_to_task:
                            f.cancel()
                        break
                        
            except Exception as e:
                print(f"❌ Task failed with exception: {e}")
                failed_count += 1
                if not args.continue_on_error:
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
        print(f"Average time per checkpoint: {avg_time:.1f}s")
    
    # Save detailed results
    results_file = os.path.join(args.output_dir, "batch_inference_results.json")
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
        print(f"\n❌ {failed_count} checkpoints failed to process")
        return 1
    else:
        print(f"\n✅ All checkpoints processed successfully!")
        return 0

if __name__ == "__main__":
    exit(main())
