#!/usr/bin/env python3
"""
run_rllm_from_parquet.py

This script faithfully adapts the original, working `run_deepcoder.py` to
meet the user's requirements.

1.  It loads a user-specified Parquet file.
2.  It uses a `preprocess_fn` that is a direct, faithful replication of the
    logic in `prepare_deepcoder_data.py` to format the data correctly.
3.  It uses the standard `SingleTurnEnvironment` as in the original script.
4.  It executes tasks in chunks to enable incremental saving.
5.  It filters for trajectories with a reward of 1 and saves them at the
    specified interval.

This is the definitive, correct implementation.

Usage:
    python examples/deepcoder/run_rllm_from_parquet.py \
        --input-file /path/to/train.parquet \
        --output-dir ./validated_trajectories
"""
import argparse
import asyncio
import os
import json
import math
from datetime import datetime

# --- RLLM Framework Imports (from original script) ---
from rllm.agents.code_agent import CompetitionCodingAgent
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.base.single_turn_env import SingleTurnEnvironment
from rllm.rewards.reward_fn import code_reward_fn
from rllm.utils import save_trajectories
# Import the key function from the data preparation script
from rllm.data.utils import fetch_live_code_bench_system_prompt

# --- Third-Party Library Imports ---
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# Ensure TOKENIZERS_PARALLELISM is set
os.environ["TOKENIZERS_PARALLELISM"] = "true"


# --- STEP 1: FAITHFULLY REPLICATE THE PREPROCESSING LOGIC ---
# This function is a direct adaptation of the `preprocess_fn` from
# `prepare_deepcoder_data.py`. This is the key to formatting the data correctly.
def preprocess_fn(example, idx):
    starter_code = example.get("starter_code", "")
    question = fetch_live_code_bench_system_prompt(example["problem"], starter_code if starter_code else None)

    tests_raw = example["tests"]
    if isinstance(tests_raw, str):
        tests = json.loads(tests_raw)
    else:
        tests = tests_raw
    # Metadata in the parquet is a JSON string (e.g., "null"). Parse to dict.
    metadata_raw = example.get("metadata", {})
    if isinstance(metadata_raw, str):
        try:
            metadata = json.loads(metadata_raw)
        except Exception:
            metadata = {}
    else:
        metadata = metadata_raw or {}

    if isinstance(tests, dict) and "inputs" in tests and "outputs" in tests:
        normalized_tests = []
        for input_val, output_val in zip(tests["inputs"], tests["outputs"], strict=False):
            normalized_tests.append({"input": input_val, "output": output_val, "testtype": "stdin_stdout"})
        tests = normalized_tests
    
    if not isinstance(tests, list):
        tests = [tests] if tests else []

    # The original script does a deeper test metadata check which we replicate
    # here for full fidelity, though it may not be strictly necessary for all cases.
    for test in tests:
        if test.get("testtype") == "functional" and metadata.get("func_name") is not None:
            test["metadata"] = {"func_name": str(metadata["func_name"])}
        else:
            test["metadata"] = {"func_name": None}

    # This is the exact dictionary structure the engine expects for each task.
    return {
        "question": question,
        "ground_truth": json.dumps(tests), # Ground truth is the tests
        "data_source": "livecodebench",
        "uid": f"deepcoder_{idx}",
        "index": idx,
        "starter_code": starter_code,
        "metadata": json.dumps(metadata)
    }

def main():
    parser = argparse.ArgumentParser(
        description="Run RLLM agent from Parquet with incremental saving."
    )
    parser.add_argument("--input-file", required=True, help="Path to the input Parquet file.")
    parser.add_argument("--output-dir", required=True, help="Directory to save the output trajectory files.")
    parser.add_argument("--save-interval", type=int, default=1000, help="Save after N successful (reward=1) trajectories.")
    parser.add_argument("--n-parallel-agents", type=int, default=64, help="Number of parallel agents.")
    parser.add_argument("--chunk-multiplier", type=int, default=4, help="Factor to determine chunk size.")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume by skipping UIDs already saved in output-dir.")
    parser.add_argument("--skip-existing", action="store_true", default=True, help="Skip tasks whose UIDs already exist in saved trajectory files.")
    parser.add_argument("--prefilter-overlong", action="store_true", default=True, help="Skip tasks whose initial prompt exceeds max_prompt_length.")
    parser.add_argument("--save-mode", choices=["chunk", "interval"], default="chunk", help="Save successes every chunk or when reaching save-interval.")
    parser.add_argument("--failed-dir", type=str, default=None, help="Directory to save failed trajectories and tasks (default: <output-dir>/failed)")
    parser.add_argument("--save-failed", action="store_true", default=True, help="Save executed but failed (reward<=0) trajectories")
    parser.add_argument("--save-failed-tasks", action="store_true", default=True, help="Save tasks that could not execute at all")
    args = parser.parse_args()

    # --- STEP 2: INITIALIZE THE ENGINE (from original script) ---
    print("Initializing RLLM AgentExecutionEngine...")
    model_name = "agentica-org/DeepCoder-14B-Preview"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    reward_fn = code_reward_fn

    env_args = {"reward_fn": reward_fn}
    sampling_params = {"temperature": 0.6, "top_p": 0.95, "model": model_name}

    engine = AgentExecutionEngine(
        agent_class=CompetitionCodingAgent,
        env_class=SingleTurnEnvironment,
        agent_args={},
        env_args=env_args,
        engine_name="openai",
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        rollout_engine_args={
            "base_url": "http://localhost:30000/v1",
            "api_key": "None",
        },
        max_response_length=32768,
        max_prompt_length=4096,
        n_parallel_agents=args.n_parallel_agents,
        max_workers=args.n_parallel_agents,
    )
    print("Engine initialized.")

    # --- STEP 3: LOAD AND PREPROCESS THE DATA (New Logic) ---
    print(f"Loading dataset from {args.input_file}...")
    dataset = load_dataset("parquet", data_files=args.input_file, split="train")
    
    print("Preprocessing and formatting data...")
    # Use .map() to apply our faithful preprocessing function to every example
    processed_dataset = dataset.map(preprocess_fn, with_indices=True, num_proc=os.cpu_count())
    
    # .get_data() is not a method of HF datasets, so we just use the processed dataset
    tasks = list(processed_dataset)
    print(f"Created {len(tasks)} tasks.")

    # --- STEP 3.1: Resume support by skipping already-saved UIDs ---
    def load_existing_uids(save_dir: str) -> set[str]:
        import glob
        import torch
        from rllm.agents.agent import Trajectory  # noqa: F401  # for unpickler

        uids: set[str] = set()
        for pt in glob.glob(os.path.join(save_dir, "*.pt")):
            try:
                data = torch.load(pt, map_location="cpu", weights_only=False)
            except TypeError:
                data = torch.load(pt, map_location="cpu")
            if isinstance(data, (list, tuple)):
                for traj in data:
                    try:
                        task = getattr(traj, "task", None)
                        if isinstance(task, dict) and task.get("uid"):
                            uids.add(task["uid"])
                    except Exception:
                        continue
        return uids

    existing_uids = set()
    if args.resume or args.skip_existing:
        os.makedirs(args.output_dir, exist_ok=True)
        existing_uids = load_existing_uids(args.output_dir)
        # Also include failed-dir saved UIDs
        failed_dir = args.failed_dir or os.path.join(args.output_dir, "failed")
        os.makedirs(failed_dir, exist_ok=True)
        def load_failed_uids(fdir: str) -> set[str]:
            import glob
            import torch
            from rllm.agents.agent import Trajectory  # noqa: F401
            u: set[str] = set()
            # from failed trajectory pt files
            for pt in glob.glob(os.path.join(fdir, "failed_trajectories_*.pt")):
                try:
                    data = torch.load(pt, map_location="cpu", weights_only=False)
                except TypeError:
                    data = torch.load(pt, map_location="cpu")
                if isinstance(data, (list, tuple)):
                    for traj in data:
                        try:
                            task = getattr(traj, "task", None)
                            if isinstance(task, dict) and task.get("uid"):
                                u.add(task["uid"])
                        except Exception:
                            continue
            # from failed task jsonl
            for jf in glob.glob(os.path.join(fdir, "failed_tasks_*.jsonl")):
                try:
                    with open(jf, "r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                obj = json.loads(line)
                                uid = obj.get("uid")
                                if isinstance(uid, str):
                                    u.add(uid)
                            except Exception:
                                continue
                except Exception:
                    continue
            return u
        failed_uids = load_failed_uids(failed_dir)
        if failed_uids:
            print(f"Resume: also skipping {len(failed_uids)} previously failed UIDs.")
            existing_uids |= failed_uids
        if existing_uids:
            print(f"Resume: skipping {len(existing_uids)} UIDs already saved.")

    if existing_uids:
        before = len(tasks)
        tasks = [t for t in tasks if t.get("uid") not in existing_uids]
        print(f"Filtered tasks by resume: {before} -> {len(tasks)}")

    # --- STEP 3.2: Prefilter tasks whose initial prompt exceeds max length ---
    if args.prefilter_overlong:
        def prompt_len(task: dict) -> int:
            # Build minimal messages list (single user message) and tokenize via chat template
            messages = [{"role": "user", "content": str(task.get("question", ""))}]
            prompt_text = engine.chat_parser.parse(messages, add_generation_prompt=True, is_first_msg=True)
            return len(tokenizer.encode(prompt_text, add_special_tokens=False))

        filtered = []
        skipped = 0
        for t in tasks:
            try:
                pl = prompt_len(t)
            except Exception:
                pl = 10**9
            if pl > engine.max_prompt_length:
                print(f"Skip overlong UID={t.get('uid')} prompt_len={pl} > {engine.max_prompt_length}")
                skipped += 1
                continue
            filtered.append(t)
        if skipped:
            print(f"Overlong prefilter: skipped {skipped} tasks; remaining {len(filtered)}")
        tasks = filtered

    # --- STEP 4: EXECUTE IN CHUNKS AND SAVE (New Logic) ---
    successful_trajectories, total_saved_count, total_failed_count = [], 0, 0
    chunk_size = args.n_parallel_agents * args.chunk_multiplier
    num_chunks = math.ceil(len(tasks) / chunk_size)

    print(f"\nStarting execution with {args.n_parallel_agents} parallel agents.")
    print(f"Processing {len(tasks)} tasks in {num_chunks} chunks of size {chunk_size}...")

    # Robust chunk executor with recursive bisect to isolate failures
    def execute_safe(subtasks: list[dict]) -> tuple[list, list[dict]]:
        # returns (success_results, failed_tasks)
        if not subtasks:
            return [], []
        try:
            res = asyncio.run(engine.execute_tasks(subtasks))
            return res, []
        except Exception as e:
            if len(subtasks) == 1:
                uid = subtasks[0].get("uid")
                print(f"Skip failing UID={uid}: {e}")
                return [], [subtasks[0]]
            mid = len(subtasks)//2
            left_res, left_fail = execute_safe(subtasks[:mid])
            right_res, right_fail = execute_safe(subtasks[mid:])
            return left_res + right_res, left_fail + right_fail

    failed_saved_count = 0
    for i in tqdm(range(num_chunks), desc="Processing Chunks"):
        task_chunk = tasks[i*chunk_size : (i+1)*chunk_size]
        if not task_chunk: continue

        results_chunk_list, failed_tasks = execute_safe(task_chunk)

        successes_in_chunk = [traj for traj in results_chunk_list if getattr(traj, 'reward', 0) > 0]
        failures_in_chunk = len(results_chunk_list) - len(successes_in_chunk)
        failed_trajs_in_chunk = [traj for traj in results_chunk_list if getattr(traj, 'reward', 0) <= 0]
        
        successful_trajectories.extend(successes_in_chunk)
        total_failed_count += failures_in_chunk
        # Count skipped failures
        total_failed_count += len(failed_tasks)

        # Save policy
        should_save = False
        if args.save_mode == "interval" and len(successful_trajectories) >= args.save_interval:
            should_save = True
        if args.save_mode == "chunk" and successes_in_chunk:
            should_save = True

        if should_save:
            # Generate incremental filename with counter rather than count-based to avoid overwriting or confusion
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            counter = total_saved_count + len(successful_trajectories)
            filename = f"trajectories_reward1_{counter}_{timestamp}.pt"
            print(f"\nSaving {len(successful_trajectories)} successful trajectories (mode={args.save_mode})...")
            try:
                save_trajectories(successful_trajectories, save_dir=args.output_dir, filename=filename)
                total_saved_count += len(successful_trajectories)
                successful_trajectories.clear()
            except Exception as e:
                print(f"Warning: failed to save trajectories this round: {e}")

        # Save failed trajectories and failed tasks per chunk
        if args.save_failed and failed_trajs_in_chunk:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            failed_dir = args.failed_dir or os.path.join(args.output_dir, "failed")
            os.makedirs(failed_dir, exist_ok=True)
            counter = failed_saved_count + len(failed_trajs_in_chunk)
            fname = f"failed_trajectories_{counter}_{timestamp}.pt"
            try:
                save_trajectories(failed_trajs_in_chunk, save_dir=failed_dir, filename=fname)
                failed_saved_count += len(failed_trajs_in_chunk)
            except Exception as e:
                print(f"Warning: failed to save failed trajectories: {e}")

        if args.save_failed_tasks and failed_tasks:
            from pathlib import Path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            failed_dir = args.failed_dir or os.path.join(args.output_dir, "failed")
            os.makedirs(failed_dir, exist_ok=True)
            jsonl_path = Path(failed_dir) / f"failed_tasks_{timestamp}.jsonl"
            try:
                with open(jsonl_path, "w", encoding="utf-8") as f:
                    for t in failed_tasks:
                        rec = {"uid": t.get("uid"), "note": "execution_failed"}
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Warning: failed to save failed tasks: {e}")

    if successful_trajectories:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        counter = total_saved_count + len(successful_trajectories)
        filename = f"trajectories_reward1_{counter}_{timestamp}.pt"
        print(f"\nSaving final batch of {len(successful_trajectories)} successful trajectories...")
        try:
            save_trajectories(successful_trajectories, save_dir=args.output_dir, filename=filename)
            total_saved_count += len(successful_trajectories)
        except Exception as e:
            print(f"Warning: failed to save final trajectories: {e}")

    print("\n--- Processing Complete ---")
    print(f"Total successful (reward=1) trajectories saved: {total_saved_count}")
    print(f"Total failed/ignored (reward=0 or skipped): {total_failed_count}")
    print(f"Validated trajectories saved in: {args.output_dir}")

if __name__ == "__main__":
    main()