#!/usr/bin/env python3
"""
convert_trajectories_to_cots.py

Converts saved RLLM trajectories (.pt) and/or parquet task files into a
Codeforces-COTS-like Arrow dataset suitable for training.

Output format (HuggingFace Dataset saved via save_to_disk):
- Each record contains at minimum:
  - id: unique string per sample
  - question: original problem statement / prompt
  - generation: model code as a markdown fenced python block
  - public_tests: dict with keys 'input' (list[str]), 'output' (list[str])
  - private_tests: optional dict with same structure (empty by default)
  - reward: float (if available for .pt trajectories)

Notes:
- Ground-truth tests in saved RLLM tasks are usually in LiveCodeBench style
  (list[dict] with keys: type/testtype, input, output). We convert stdin_stdout
  tests into the Codeforces-COTS schema: {'input': [...], 'output': [...]}.
- Function-call style tests are ignored here to keep a consistent I/O schema.
- Multiple input files are supported (directory or glob); both .pt and .parquet
  are accepted and merged.

Usage:
  python scripts/convert_trajectories_to_cots.py \
    --input-path /path/to/dir_or_file \
    --output-path /path/to/output_arrow_dir \
    [--glob "*.pt"] [--min-reward 1.0]
"""

import argparse
import json
import os
import re
import sys
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from datasets import Dataset

# Ensure the Trajectory class is importable for torch.load deserialization
try:
    from rllm.agents.agent import Trajectory  # noqa: F401  # Needed by torch.load
except Exception:
    # If import fails, torch.load may still work with weights_only=False on simple objects
    pass


def extract_python_code(generation_str: str) -> Optional[str]:
    """Extract Python code from a markdown code block."""
    if not isinstance(generation_str, str):
        return None
    match = re.search(r"```python\n(.*?)```", generation_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    # If no fenced block, best-effort: treat entire string as code
    stripped = generation_str.strip()
    return stripped if stripped else None


def to_fenced_python(code: str) -> str:
    code = code or ""
    return f"```python\n{code}\n```"


def synthesize_thought(problem_text: str) -> str:
    """Create a short generic thinking preamble."""
    return (
        "I will parse the input exactly as specified, analyze the constraints to choose an efficient algorithm, "
        "handle edge cases carefully, then implement a clean Python solution and print the result to stdout."
    )


def deepcoder_to_mot(user_prompt_text: str) -> str:
    """Convert DeepCoder-style prompt text to Mixture-of-Thought style.

    Steps:
    1) Strip DeepCoder boilerplate header and trailer.
    2) Map '-----Input-----', '-----Output-----', '-----Examples-----', '-----Note-----' to markdown headers.
    3) Convert example pairs 'Input\n..\n\nOutput\n..' into fenced blocks ```input ...``` and ```output ...```.
    4) Prepend the MoT preamble and a '# Problem' header.
    """
    import re as _re

    s = str(user_prompt_text or "")

    # 1) Remove DeepCoder header and footer boilerplate
    s = _re.sub(
        r"^\s*You are an expert Python programmer[\s\S]*?Solve the following coding problem using the programming language python:\s*\n",
        "",
        s,
        flags=_re.IGNORECASE,
    )
    s = _re.sub(
        r"\n\s*The input will be stdin[\s\S]*$",
        "",
        s,
        flags=_re.IGNORECASE,
    )

    # 2) Replace section markers to markdown
    replacements = {
        "-----Input-----": "## Input Format",
        "-----Output-----": "## Output Format",
        "-----Examples-----": "## Examples",
        "-----Note-----": "## Note",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)

    # Ensure a blank line after the Examples heading so fences don't stick to it
    s = _re.sub(r"(?m)^## Examples[ \t]*\r?\n?", "## Examples\n\n", s)

    # 3) Convert example Input/Output blocks into fenced code blocks
    def repl_examples(match: _re.Match) -> str:
        inp = match.group(1).strip("\n")
        out = match.group(2).strip("\n")
        return f"```input\n{inp}\n```\n```output\n{out}\n```"

    # Replace repeated Input/Output pairs under Examples
    s = _re.sub(
        r"(?:^|\n)Input\s*\n([\s\S]*?)\n\s*Output\s*\n([\s\S]*?)(?=\n\n|$)",
        repl_examples,
        s,
        flags=_re.IGNORECASE,
    )

    # 4) Prepend Mixture-of-Thought preamble
    mot_preamble = (
        "You will be given a competitive programming problem.\n"
        "Analyze the maximum input constraints and identify the optimal algorithmic approach and data structures needed to process the largest possible test cases within the time and memory limits, then explain why your chosen implementation strategy is the most efficient solution. Please reason step by step about your solution approach, then provide a complete implementation in Python 3 that is thoroughly optimized for both speed and memory usage.\n\n"
        "Your solution must read input from standard input (input()), write output to standard output (print()).\n"
        "Do not include any debug prints or additional output.\n\n"
        "Put your final solution within a single code block:\n"
        "```python\n<your code here>\n```\n\n"
    )

    s = s.strip()
    if not s.startswith("# Problem"):
        s = "# Problem\n\n" + s

    return mot_preamble + s


def parse_tests_from_ground_truth(ground_truth: Any) -> Tuple[List[str], List[str]]:
    """Parse inputs/outputs from various ground_truth formats.

    Supports:
    - JSON string encoding list[dict]
    - list[dict] with keys: 'type' or 'testtype', 'input', 'output'
    - dict with 'inputs' and 'outputs' (TACO-style) -> convert to LCB-like

    Returns (inputs, outputs) both as list[str]. Non-stdin tests are ignored.
    """
    if ground_truth is None:
        return [], []

    tests = ground_truth
    if isinstance(tests, str):
        try:
            tests = json.loads(tests)
        except Exception:
            return [], []

    # TACO-style dict
    if isinstance(tests, dict) and "inputs" in tests and "outputs" in tests:
        ins = tests.get("inputs") or []
        outs = tests.get("outputs") or []
        # stringify to be consistent
        return [str(x) for x in ins], [str(x) for x in outs]

    # LCB-style list
    if isinstance(tests, list):
        inputs: List[str] = []
        outputs: List[str] = []
        for t in tests:
            if not isinstance(t, dict):
                continue
            ttype = t.get("type") or t.get("testtype")
            if ttype != "stdin_stdout":
                continue
            if "input" in t and "output" in t:
                inputs.append(str(t["input"]))
                outputs.append(str(t["output"]))
        return inputs, outputs

    return [], []


STRIP_POST_THINK: bool = False  # set in main via CLI


def iter_pt_trajectories(pt_path: str) -> List[Dict[str, Any]]:
    """Load a .pt file and convert contained trajectories into training rows."""
    try:
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
    except TypeError:
        # For older torch versions without weights_only
        data = torch.load(pt_path, map_location="cpu")

    rows: List[Dict[str, Any]] = []

    # The saved object is usually a list of Trajectory or list-like
    if not isinstance(data, (list, tuple)):
        data = [data]

    for idx, traj in enumerate(data):
        try:
            # Try accessing RLLM Trajectory API
            task = getattr(traj, "task", None) or {}
            steps = getattr(traj, "steps", [])
            reward = float(getattr(traj, "reward", 0.0))

            # Original task info
            question = None
            uid = None
            ground_truth = None
            if isinstance(task, dict):
                question = task.get("question") or task.get("prompt") or None
                uid = task.get("uid")
                ground_truth = task.get("ground_truth") or task.get("tests")

            # Response/code: last step model_response preferred, else action
            code_text = None
            if steps:
                last = steps[-1]
                code_text = getattr(last, "model_response", None) or getattr(last, "action", None)
                if isinstance(code_text, dict):
                    code_text = json.dumps(code_text)
                if code_text is None and hasattr(last, "chat_completions"):
                    # fallback to assistant content
                    msgs = getattr(last, "chat_completions", [])
                    for m in reversed(list(msgs)):
                        if isinstance(m, dict) and m.get("role") == "assistant":
                            code_text = m.get("content")
                            break

            user_prompt = question or ""
            # If no code_text, skip this sample
            if not code_text or not str(code_text).strip():
                continue
            # Extract or use raw code
            code_src = extract_python_code(code_text) or str(code_text).strip()
            if not code_src: # Robust, probably not needed as we get reward 1.0
                continue
            # Erran: strange, DeepCoder does not have <think> in front of code_text! A bug needs to be fixed. TODO
            assistant_answer = f"<think>\n{code_text}"
            generation = assistant_answer

            # Optional: strip everything after </think> except the solution code
            if STRIP_POST_THINK:
                try:
                    ga = generation
                    if isinstance(ga, str) and "</think>" in ga:
                        pre = ga.split("</think>", 1)[0] + "</think>\n"
                        generation = pre + to_fenced_python(code_src)
                        assistant_answer = generation
                except Exception:
                    pass

            mot_user_prompt = deepcoder_to_mot(user_prompt)

            messages = [
                {"role": "user", "content": mot_user_prompt},
                {"role": "assistant", "content": assistant_answer},
            ]

            ins, outs = parse_tests_from_ground_truth(ground_truth)
            public_tests = {"input": [str(x) for x in ins], "output": [str(x) for x in outs]}


            row = {
                "id": uid or f"pt:{os.path.basename(pt_path)}#{idx}",
                "question": mot_user_prompt,
                "prompt": mot_user_prompt,
                "generation": generation,
                "messages": messages,
                "public_tests": public_tests,
                "private_tests": {"input": [], "output": []},
                "reward": reward,
                "source_file": os.path.basename(pt_path),
            }
            rows.append(row)
        except Exception:
            # Skip malformed entries
            continue

    return rows


def collect_input_files(input_path: str, pattern: Optional[str]) -> List[str]:
    if os.path.isdir(input_path):
        pat = pattern or "*"
        return sorted(glob(os.path.join(input_path, pat)))
    if any(char in input_path for char in "*?[]"):
        return sorted(glob(input_path))
    return [input_path]

"""
python /workspace/rllm/scripts/convert_trajectories_to_cots.py \
  --input-path /mnt/data2/deepcoder_trajectories_sample \
  --glob "*.pt" \
  --output-path /mnt/data2/tmp \
  --min-reward 1.0

  python /workspace/rllm/scripts/convert_trajectories_to_cots.py \
  --input-path /mnt/data2/deepcoder_trajectories \
  --glob "*.pt" \
  --output-path /mnt/data2/new_deepcoder_cots_arrow_appexp \
  --min-reward 1.0

  python /workspace/rllm/scripts/convert_trajectories_to_cots.py \
  --input-path /mnt/data2/deepcoder_trajectories \
  --glob "*.pt" \
  --output-path /mnt/data2/deepcoder_cots_arrow_codeonly \
  --min-reward 1.0 \
  --strip-post-think
"""
def main():
    parser = argparse.ArgumentParser(description="Convert RLLM trajectories and parquet files to COTS-like Arrow dataset.")
    parser.add_argument("--input-path", default="/mnt/data2/deepcoder_trajectories", help="Directory, glob, or file path to .pt/.parquet inputs")
    parser.add_argument("--output-path", default="/mnt/data2/new_deepcoder_cots_arrow", help="Output directory for HF Arrow dataset (save_to_disk)")
    parser.add_argument("--glob", dest="glob_pattern", default=None, help="Optional glob pattern when input-path is a directory, e.g., '*.pt' or '*.parquet'")
    parser.add_argument("--min-reward", type=float, default=None, help="If set, filter .pt trajectories by minimum reward")
    parser.add_argument("--strip-post-think", action="store_true", help="If set, strip everything after </think> and keep only the fenced solution code")
    args = parser.parse_args()

    files = collect_input_files(args.input_path, args.glob_pattern)

    # Set global option
    global STRIP_POST_THINK
    STRIP_POST_THINK = bool(args.strip_post_think)
    if not files:
        print("No input files found.")
        sys.exit(1)

    all_rows: List[Dict[str, Any]] = []
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext == ".pt":
            rows = iter_pt_trajectories(f)
            if args.min_reward is not None:
                rows = [r for r in rows if r.get("reward") is not None and float(r.get("reward", 0.0)) >= args.min_reward]
            all_rows.extend(rows)
        elif ext == ".parquet":
            assert False, "Parquet files are not supported yet"
        else:
            # ignore unknown files
            continue

    if not all_rows:
        print("No convertible rows found from the provided inputs.")
        sys.exit(1)

    # Sanitize rows to strictly match schema
    def sanitize_row(r: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        out["id"] = str(r.get("id", ""))
        q = r.get("question") or r.get("prompt") or ""
        g = r.get("generation") or ""
        out["question"] = str(q)
        out["prompt"] = str(q)
        out["generation"] = str(g)

        # messages: prefer existing valid messages; else build with assistant content = generation (preserve <think>)
        msgs = r.get("messages")
        if isinstance(msgs, list) and len(msgs) >= 2 and isinstance(msgs[0], dict) and isinstance(msgs[1], dict):
            out["messages"] = [
                {"role": str(msgs[0].get("role", "user")), "content": str(msgs[0].get("content", out["question"]))},
                {"role": str(msgs[1].get("role", "assistant")), "content": str(msgs[1].get("content", out["generation"]))},
            ]
        else:
            out["messages"] = [
                {"role": "user", "content": str(out["question"])},
                {"role": "assistant", "content": str(out["generation"])},
            ]

        # tests: ensure dict with string lists
        def _to_str_list(x):
            if isinstance(x, list):
                return [str(e) for e in x]
            return []
        pt = r.get("public_tests") or {}
        out["public_tests"] = {
            "input": _to_str_list(pt.get("input")),
            "output": _to_str_list(pt.get("output")),
        }
        pr = r.get("private_tests") or {}
        out["private_tests"] = {
            "input": _to_str_list(pr.get("input")),
            "output": _to_str_list(pr.get("output")),
        }

        reward = r.get("reward")
        try:
            out["reward"] = float(reward) if reward is not None else 0.0
        except Exception:
            out["reward"] = 0.0
        out["source_file"] = str(r.get("source_file", ""))
        return out

    all_rows = [sanitize_row(r) for r in all_rows]

    print(f"Collected {len(all_rows)} examples. Saving to {args.output_path} ...")
    # Rely on schema inference to avoid nested Features pitfalls
    ds = Dataset.from_list(all_rows)
    ds.save_to_disk(args.output_path)
    print("Done.")


if __name__ == "__main__":
    main()


