#!/usr/bin/env python3
"""
Batch evaluation script for processing multiple checkpoint solutions.

This script evaluates solutions from multiple checkpoints in parallel, generating
benchmark results for each checkpoint and saving them to organized directories.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch evaluation for multiple checkpoint solutions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all checkpoint solutions
  python eval/batch_eval_solutions.py \\
    --solutions-dir /mnt/data3/model_solution_grad_norm10_16k_wu005_lr5e05_30epoch \\
    --output-dir /mnt/data3/eval_results_grad_norm10_16k_wu005_lr5e05_30epoch

  # Evaluate specific checkpoints
  python eval/batch_eval_solutions.py \\
    --solutions-pattern "/mnt/data3/solutions/checkpoint-{196,392,588}" \\
    --output-dir /mnt/data3/selected_eval_results

  # Custom evaluation settings
  python eval/batch_eval_solutions.py \\
    --solutions-dir /mnt/data3/solutions \\
    --output-dir /mnt/data3/eval_results \\
    --max-generated-tests 5 \\
    --workers 32
        """
    )
    
    # Input specification
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--solutions-dir",
        help="Directory containing checkpoint solution directories"
    )
    input_group.add_argument(
        "--solutions-pattern",
        help="Glob pattern for solution directories"
    )
    input_group.add_argument(
        "--solutions-list",
        help="File containing list of solution directories (one per line)"
    )
    
    # Output and processing options
    parser.add_argument(
        "--output-dir", required=True,
        help="Base output directory for evaluation results"
    )
    parser.add_argument(
        "--max-workers", type=int, default=4,
        help="Maximum number of parallel evaluation workers"
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--max-generated-tests", type=int, default=0,
        help="Maximum number of generated tests (default: 0)"
    )
    parser.add_argument(
        "--generated-tests-workers", type=int, default=16,
        help="Number of workers for generated tests (default: 16)"
    )
    parser.add_argument(
        "--endpoint", default="http://localhost:2000",
        help="Evaluation endpoint URL (default: http://localhost:2000)"
    )
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="Timeout per evaluation in seconds (default: 300)"
    )
    
    # Control options
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be processed without running evaluation"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing results"
    )
    parser.add_argument(
        "--continue-on-error", action="store_true",
        help="Continue processing if individual evaluations fail"
    )
    parser.add_argument(
        "--checkpoint-filter",
        help="Filter checkpoints by pattern (e.g., 'checkpoint-{196,392,588}')"
    )
    parser.add_argument(
        "--sort-order", choices=["step", "name", "none"], default="step",
        help="Sort order for processing checkpoints (default: step)"
    )
    
    return parser.parse_args()

def find_solution_directories(args) -> List[str]:
    """Find all solution directories to process."""
    solutions = []
    
    if args.solutions_dir:
        # Find all subdirectories in solutions_dir
        pattern = os.path.join(args.solutions_dir, "*")
        candidates = glob.glob(pattern)
        solutions.extend([p for p in candidates if os.path.isdir(p)])
    elif args.solutions_pattern:
        candidates = glob.glob(args.solutions_pattern)
        solutions.extend([p for p in candidates if os.path.isdir(p)])
    elif args.solutions_list:
        with open(args.solutions_list) as f:
            solutions.extend([line.strip() for line in f if line.strip()])
    
    # Apply filter if specified
    if args.checkpoint_filter:
        import fnmatch
        solutions = [s for s in solutions if fnmatch.fnmatch(os.path.basename(s), args.checkpoint_filter)]
    
    # Validate solutions have JSONL files
    valid_solutions = []
    for sol in solutions:
        sol_path = Path(sol)
        has_jsonl = any([
            list(sol_path.glob("*.jsonl")),
            list(sol_path.glob("**/*.jsonl"))
        ])
        if has_jsonl:
            valid_solutions.append(sol)
        else:
            print(f"Warning: No JSONL files found in {sol}")
    
    # Sort based on specified order
    if args.sort_order == "step":
        def solution_sort_key(solution_path):
            """Extract numeric step for proper sorting."""
            solution_name = os.path.basename(solution_path)
            
            # Try to extract step number
            if "checkpoint-" in solution_name:
                try:
                    step_str = solution_name.split("checkpoint-")[1].split("_")[0].split("-")[0]
                    return int(step_str)
                except (IndexError, ValueError):
                    pass
            
            # Fallback to alphabetical sorting
            return float('inf'), solution_name
        
        valid_solutions = sorted(valid_solutions, key=solution_sort_key)
    elif args.sort_order == "name":
        valid_solutions = sorted(valid_solutions)
    # else: "none" - keep original order
    
    return valid_solutions

def extract_checkpoint_info(solution_path: str) -> Dict[str, str]:
    """Extract checkpoint information from solution path."""
    solution_name = os.path.basename(solution_path)
    
    # Try to extract step number
    step = "unknown"
    if "checkpoint-" in solution_name:
        try:
            step = solution_name.split("checkpoint-")[1].split("_")[0].split("-")[0]
        except (IndexError, ValueError):
            pass
    
    return {
        "name": solution_name,
        "step": step,
        "path": solution_path
    }

def find_solution_files(solution_dir: str) -> List[str]:
    """Find all JSONL solution files in a directory."""
    solution_path = Path(solution_dir)
    jsonl_files = []
    
    # Look for JSONL files in the directory and subdirectories
    for pattern in ["*.jsonl", "**/*.jsonl"]:
        jsonl_files.extend(list(solution_path.glob(pattern)))
    
    return [str(f) for f in jsonl_files]

def generate_output_dir(checkpoint_info: Dict[str, str], base_output_dir: str) -> str:
    """Generate output directory name for checkpoint evaluation results."""
    if checkpoint_info["step"] != "unknown":
        return os.path.join(base_output_dir, f"eval_checkpoint_{checkpoint_info['step']}")
    else:
        return os.path.join(base_output_dir, f"eval_{checkpoint_info['name']}")

def run_evaluation_for_checkpoint(
    solution_dir: str,
    output_dir: str,
    args
) -> Dict[str, Any]:
    """Run evaluation for a single checkpoint's solutions."""
    result = {
        "solution_dir": solution_dir,
        "output_dir": output_dir,
        "success": False,
        "error": None,
        "start_time": time.time(),
        "end_time": None,
        "duration": None,
        "solution_files": [],
        "evaluation_files": []
    }
    
    try:
        # Find solution files
        solution_files = find_solution_files(solution_dir)
        if not solution_files:
            result["error"] = "No JSONL solution files found"
            print(f"❌ No solution files found in {solution_dir}")
            return result
        
        result["solution_files"] = solution_files
        print(f"🔍 Found {len(solution_files)} solution files in {os.path.basename(solution_dir)}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each solution file
        evaluation_files = []
        for solution_file in solution_files:
            solution_filename = os.path.basename(solution_file)
            output_filename = f"eval_{solution_filename}"
            output_file = os.path.join(output_dir, output_filename)
            
            # Build evaluation command
            cmd = [
                sys.executable,
                "eval/eval_with_piston_gentest_checker_stats.py",
                "--solutions-path", solution_file,
                "--max-generated-tests", str(args.max_generated_tests),
                "--generated-tests-workers", str(args.generated_tests_workers),
                "--endpoint", args.endpoint
            ]
            
            print(f"🚀 Evaluating: {solution_filename}")
            
            if args.dry_run:
                print(f"[DRY RUN] Would run: {' '.join(cmd)}")
                evaluation_files.append(output_file)
            else:
                # Run evaluation with timeout
                try:
                    process_result = subprocess.run(
                        cmd, 
                        capture_output=True, 
                        text=True, 
                        timeout=args.timeout,
                        cwd=os.getcwd()
                    )
                    
                    if process_result.returncode == 0:
                        # Save output to file
                        with open(output_file, 'w') as f:
                            f.write(process_result.stdout)
                        evaluation_files.append(output_file)
                        print(f"✅ Completed: {solution_filename}")
                    else:
                        error_msg = process_result.stderr.strip() if process_result.stderr else "No error message"
                        result["error"] = f"Evaluation failed for {solution_filename}: {error_msg}"
                        print(f"❌ Failed: {solution_filename} - {error_msg}")
                        
                        # Save error output
                        error_file = os.path.join(output_dir, f"error_{solution_filename}.txt")
                        with open(error_file, 'w') as f:
                            f.write(f"Error: {error_msg}\n")
                            f.write(f"Stdout: {process_result.stdout}\n")
                            f.write(f"Stderr: {process_result.stderr}\n")
                        evaluation_files.append(error_file)
                        
                except subprocess.TimeoutExpired:
                    result["error"] = f"Evaluation timeout for {solution_filename}"
                    print(f"⏰ Timeout: {solution_filename}")
                    
                    # Save timeout info
                    timeout_file = os.path.join(output_dir, f"timeout_{solution_filename}.txt")
                    with open(timeout_file, 'w') as f:
                        f.write(f"Evaluation timed out after {args.timeout} seconds\n")
                    evaluation_files.append(timeout_file)
        
        result["evaluation_files"] = evaluation_files
        
        # If at least one evaluation succeeded, mark as success
        if evaluation_files:
            result["success"] = True
            print(f"✅ Completed evaluation for {os.path.basename(solution_dir)}")
        else:
            result["error"] = "All evaluations failed"
            print(f"❌ All evaluations failed for {os.path.basename(solution_dir)}")
            
    except Exception as e:
        result["error"] = str(e)
        print(f"❌ Failed: {os.path.basename(solution_dir)} - {result['error']}")
    
    result["end_time"] = time.time()
    result["duration"] = result["end_time"] - result["start_time"]
    
    return result

def main():
    args = parse_args()
    
    print("=== Batch Solution Evaluation ===")
    
    # Find solution directories
    solutions = find_solution_directories(args)
    if not solutions:
        print("Error: No valid solution directories found")
        return 1
    
    print(f"Found {len(solutions)} solution directories to process")
    
    # Show processing order
    if len(solutions) > 0:
        print(f"Processing order (sorted by {args.sort_order}):")
        for i, solution in enumerate(solutions[:5], 1):  # Show first 5
            checkpoint_info = extract_checkpoint_info(solution)
            print(f"  {i:2d}. {checkpoint_info['name']} (step {checkpoint_info['step']})")
        if len(solutions) > 5:
            print(f"  ... and {len(solutions) - 5} more")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare tasks
    tasks = []
    for solution_dir in solutions:
        checkpoint_info = extract_checkpoint_info(solution_dir)
        output_dir = generate_output_dir(checkpoint_info, args.output_dir)
        
        # Check if results already exist
        if not args.force and os.path.exists(output_dir):
            eval_files = glob.glob(os.path.join(output_dir, "eval_*.txt"))
            if eval_files:
                print(f"⏭️  Skipping {checkpoint_info['name']} (results exist)")
                continue
        
        tasks.append({
            "solution_dir": solution_dir,
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
            solution_files = find_solution_files(task["solution_dir"])
            print(f"{i:3d}. {task['checkpoint_info']['name']} -> {task['output_dir']}")
            print(f"     Solution files: {len(solution_files)}")
        return 0
    
    # Process tasks with threading
    results = []
    failed_count = 0
    
    def process_task(task):
        """Process a single evaluation task."""
        result = run_evaluation_for_checkpoint(
            task["solution_dir"],
            task["output_dir"],
            args
        )
        return result
    
    # Use ThreadPoolExecutor for concurrent processing
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(process_task, task): task 
            for task in tasks
        }
        
        # Collect results
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
                
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
    results_file = os.path.join(args.output_dir, "batch_eval_results.json")
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
        print(f"\n❌ {failed_count} checkpoints failed to evaluate")
        return 1
    else:
        print(f"\n✅ All checkpoints evaluated successfully!")
        return 0

if __name__ == "__main__":
    exit(main())
