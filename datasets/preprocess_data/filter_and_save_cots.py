#!/usr/bin/env python3
"""
Filter and save high-quality Codeforces CoT solutions into Arrow.

Consolidated script (combines prior sol_py and w_editorials variants) with CLI flags:
- --subset selects which dataset subset to use (solutions_py vs solutions_w_editorials_py)
- --subset-name allows direct override of subset string
- --match-strategy selects output comparator: basic or robust (tolerant numeric)
- --include-generated-tests optionally falls back to generated tests if no public/private
- --failed-output-dir optionally saves failed samples for later analysis
- --streaming and --dataset-split control HF dataset loading
"""

import argparse
import json
import re
from typing import Optional, List, Tuple, Dict, Any
from multiprocessing import Pool, cpu_count

import requests
from datasets import load_dataset, Dataset
from tqdm import tqdm

# --- Configuration ---
DEFAULT_ENDPOINT = "http://localhost:2000"
DATASET_NAME = "open-r1/codeforces-cots"

# Map friendly subset keys to actual HF subset names
SUBSET_MAP = {
    "solutions_py": "solutions_py_decontaminated",
    "solutions_w_editorials_py": "solutions_w_editorials_py_decontaminated",
}

# Globals configured from CLI (to avoid changing worker arg shapes)
G_INCLUDE_GENERATED: bool = True
G_MATCH_STRATEGY: str = "robust"  # or "basic"
G_ABS_TOL: float = 1e-6
G_REL_TOL: float = 1e-6

# --- Output Matching Logic ---

def _normalize_text_basic(text: str) -> str:
    if text is None:
        return ""
    lines = text.splitlines()
    return "\n".join(line.rstrip() for line in lines).strip()

def outputs_match_basic(expected: str, actual: str) -> bool:
    e = _normalize_text_basic(expected)
    a = _normalize_text_basic(actual)
    return e == a

def outputs_match_robust(expected: str, actual: str, abs_tol: float = 1e-6, rel_tol: float = 1e-6) -> bool:
    # Exact normalized match fast-path
    e_raw = _normalize_text_basic(expected)
    a_raw = _normalize_text_basic(actual)
    if e_raw == a_raw:
        return True

    # Token-wise compare with numeric tolerance; case-insensitive for text
    e_tokens = e_raw.split()
    a_tokens = a_raw.split()
    if len(e_tokens) == len(a_tokens):
        def _is_number(tok: str) -> bool:
            try:
                float(tok)
                return True
            except Exception:
                return False

        for et, at in zip(e_tokens, a_tokens):
            if _is_number(et) and _is_number(at):
                try:
                    e_val = float(et)
                    a_val = float(at)
                    if abs(a_val - e_val) <= abs_tol:
                        continue
                    denom = max(1.0, abs(e_val))
                    if abs(a_val - e_val) / denom <= rel_tol:
                        continue
                    return False
                except Exception:
                    return False
            else:
                if et.lower() != at.lower():
                    return False
        return True

    # Multiset compare (case-insensitive for text; tolerant numeric greedy matching)
    from collections import Counter

    def _partition_tokens(tokens: List[str]) -> Tuple[List[float], Counter]:
        nums: List[float] = []
        texts: Counter = Counter()
        for t in tokens:
            try:
                nums.append(float(t))
            except Exception:
                texts[t.lower()] += 1
        return nums, texts

    e_nums, e_texts = _partition_tokens(e_tokens)
    a_nums, a_texts = _partition_tokens(a_tokens)
    if e_texts != a_texts:
        return False
    if len(e_nums) != len(a_nums):
        return False
    used = [False] * len(a_nums)
    for ev in e_nums:
        matched = False
        for j, av in enumerate(a_nums):
            if used[j]:
                continue
            if abs(av - ev) <= abs_tol or abs(av - ev) / max(1.0, abs(ev)) <= rel_tol:
                used[j] = True
                matched = True
                break
        if not matched:
            return False
    return True

# --- Helper Functions ---

def extract_python_code(generation_str: str) -> Optional[str]:
    """Extracts Python code from a markdown block."""
    match = re.search(r"```python\n(.*?)```", generation_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def parse_and_combine_tests(sample: dict) -> Tuple[List[Tuple[str, str]], str]:
    """
    Parses public and private tests from the sample and combines them.
    Optionally falls back to 'generated_tests' when enabled via G_INCLUDE_GENERATED.
    Returns (test_pairs, source_label)
    """
    all_tests: List[Tuple[str, str]] = []
    sources_used: set[str] = set()
    test_keys_to_check = ['public_tests', 'private_tests']

    for key in test_keys_to_check:
        test_data = sample.get(key)
        if (isinstance(test_data, dict) and
                'input' in test_data and 'output' in test_data and
                test_data['input'] and test_data['output'] and
                len(test_data['input']) == len(test_data['output'])):
            inputs = test_data['input']
            outputs = test_data['output']
            all_tests.extend(zip(inputs, outputs))
            sources_used.add(key)

    if G_INCLUDE_GENERATED and not all_tests:
        gen = sample.get('generated_tests')
        if (isinstance(gen, dict) and
                'input' in gen and 'output' in gen and
                gen['input'] and gen['output'] and
                len(gen['input']) == len(gen['output'])):
            all_tests.extend(zip(gen['input'], gen['output']))
            sources_used.add('generated_tests')

    if not sources_used:
        label = 'none'
    elif sources_used == {'generated_tests'}:
        label = 'generated_only'
    elif sources_used <= {'public_tests', 'private_tests'}:
        label = 'public_private'
    else:
        label = 'mixed'

    return all_tests, label


def validate_sample(sample_and_endpoint: tuple[dict, str]) -> Dict[str, Any]:
    """
    Worker function to validate a single solution sample against all its tests.
    Returns a structured result dict with pass/fail, reason, and debug info.
    """
    sample, endpoint = sample_and_endpoint

    def make_id(rec: dict) -> str:
        return rec.get('id') or f"{rec.get('contest_id', '')}/{rec.get('index', '')}".strip('/') or rec.get('problem_id', '') or '<unknown>'

    sample_id = make_id(sample)
    code_to_run = extract_python_code(sample.get('generation', ''))
    if not code_to_run:
        return {"ok": False, "reason": "no_code_extracted", "sample": sample, "code": None, "sample_id": sample_id}

    all_test_cases, tests_source_label = parse_and_combine_tests(sample)
    if not all_test_cases:
        return {"ok": False, "reason": "no_valid_tests", "sample": sample, "code": code_to_run, "sample_id": sample_id}

    session = requests.Session()

    for case_i, (test_input, expected_output) in enumerate(all_test_cases):
        payload = {
            "language": "python", "version": "*",
            "files": [{"content": code_to_run}],
            "stdin": test_input, "run_timeout": 5000,
        }

        try:
            response = session.post(f"{endpoint}/api/v2/execute", json=payload, timeout=10)
            if response.status_code != 200:
                return {"ok": False, "reason": "http_error", "status_code": response.status_code, "text": response.text[:500], "sample": sample, "code": code_to_run, "sample_id": sample_id}

            result = response.json()
            run_info = result.get("run", {})

            if run_info.get("code", 0) != 0:
                return {
                    "ok": False,
                    "reason": "runtime_error",
                    "case_index": case_i,
                    "stderr": (run_info.get("stderr", "") or "")[:1000],
                    "stdout": (run_info.get("stdout", "") or "")[:1000],
                    "sample": sample,
                    "code": code_to_run,
                    "sample_id": sample_id,
                }

            actual_output = run_info.get("stdout", "")
            matched = (
                outputs_match_robust(expected_output, actual_output, G_ABS_TOL, G_REL_TOL)
                if G_MATCH_STRATEGY == 'robust' else
                outputs_match_basic(expected_output, actual_output)
            )
            if not matched:
                return {
                    "ok": False,
                    "reason": "outputs_mismatch_generated" if tests_source_label == 'generated_only' else "outputs_mismatch",
                    "case_index": case_i,
                    "input": test_input[:1000],
                    "expected": str(expected_output)[:1000],
                    "actual": str(actual_output)[:1000],
                    "sample": sample,
                    "code": code_to_run,
                    "sample_id": sample_id,
                }

        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            return {"ok": False, "reason": "exception", "error": str(e)[:500], "sample": sample, "code": code_to_run, "sample_id": sample_id}

    # If the loop completes without returning, all tests passed
    return {"ok": True, "sample": sample, "code": code_to_run, "sample_id": sample_id, "num_tests": len(all_test_cases)}

# --- Main Execution Logic ---

def main():
    global G_INCLUDE_GENERATED, G_MATCH_STRATEGY, G_ABS_TOL, G_REL_TOL

    parser = argparse.ArgumentParser(description="Filter codeforces-cots dataset for high-quality training samples.")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="URL of your local Piston API server.")
    parser.add_argument("--output-path", default="./codeforces_cots_high_quality_arrow", help="Path to save the filtered Arrow dataset.")
    parser.add_argument("--num-workers", type=int, default=cpu_count(), help="Number of parallel processes to use.")
    parser.add_argument("--subset", default="solutions_py", choices=list(SUBSET_MAP.keys()), help="Which dataset subset mapping to use.")
    parser.add_argument("--subset-name", default=None, help="Override exact HF subset name (takes precedence over --subset).")
    parser.add_argument("--dataset-split", default="train", help="HF split spec, e.g., 'train' or 'train[:200]'.")
    parser.add_argument("--streaming", action="store_true", default=False, help="Use streaming dataset loading.")
    parser.add_argument("--match-strategy", choices=["basic","robust"], default="robust", help="Output matching strategy.")
    parser.add_argument("--abs-tol", type=float, default=1e-6, help="Absolute tolerance for numeric compare (robust mode).")
    parser.add_argument("--rel-tol", type=float, default=1e-6, help="Relative tolerance for numeric compare (robust mode).")
    parser.add_argument("--include-generated-tests", action="store_true", default=True, help="Fallback to generated tests if no public/private.")
    parser.add_argument("--failed-output-dir", default=None, help="If set, save failed samples to this directory (Arrow save_to_disk).")
    parser.add_argument("--debug-print", action="store_true", default=False, help="Print detailed debug for every sample.")
    args = parser.parse_args()

    # Apply CLI to globals
    G_INCLUDE_GENERATED = bool(args.include_generated_tests)
    G_MATCH_STRATEGY = args.match_strategy
    G_ABS_TOL = float(args.abs_tol)
    G_REL_TOL = float(args.rel_tol)

    subset_name = args.subset_name or SUBSET_MAP.get(args.subset, list(SUBSET_MAP.values())[0])
    print(f"Loading dataset '{DATASET_NAME}' subset '{subset_name}' (split={args.dataset_split}, streaming={args.streaming})...")
    if args.streaming:
        dataset = load_dataset(DATASET_NAME, subset_name, split=args.dataset_split, streaming=True)
        all_samples = list(dataset)
    else:
        dataset = load_dataset(DATASET_NAME, subset_name, split=args.dataset_split)
        all_samples = list(dataset)

    total_samples = len(all_samples)
    print(f"Loaded {total_samples} samples.")

    tasks = [(sample, args.endpoint) for sample in all_samples]
    print(f"\nStarting validation using {args.num_workers} parallel workers...")
    
    passed_samples = []
    failure_counts: Dict[str, int] = {}
    failed_samples: List[Dict[str, Any]] = []
    
    def print_sample_debug(result: Dict[str, Any]) -> None:
        MAX_CODE = 2000
        code = result.get('code') or ''
        code_print = code if len(code) <= MAX_CODE else code[:MAX_CODE] + f"... [truncated {len(code) - MAX_CODE} chars]"
        sid = result.get('sample_id', '<unknown>')
        if result.get('ok'):
            print(f"\n=== SAMPLE PASSED === id={sid} tests={result.get('num_tests')}\n--- CODE ---\n{code_print}\n")
        else:
            reason = result.get('reason')
            print(f"\n=== SAMPLE FAILED === id={sid} reason={reason}")
            if reason == 'http_error':
                print(f"status_code={result.get('status_code')} text={result.get('text')}")
            elif reason == 'runtime_error':
                print(f"case_index={result.get('case_index')} stderr={result.get('stderr')}\nstdout={result.get('stdout')}")
            elif reason == 'outputs_mismatch':
                print(f"case_index={result.get('case_index')}\ninput={result.get('input')}\nexpected={result.get('expected')}\nactual={result.get('actual')}")
            elif reason == 'outputs_mismatch_generated':
                print(f"case_index={result.get('case_index')}\ninput={result.get('input')}\nexpected={result.get('expected')}\nactual={result.get('actual')}\n(note: failed only on generated_tests)")
            elif reason in ('no_code_extracted', 'no_valid_tests', 'exception'):
                extra = {k: v for k, v in result.items() if k not in {'ok','sample','code','sample_id','reason'}}
                if extra:
                    print(json.dumps(extra, ensure_ascii=False))
            print(f"--- CODE ---\n{code_print}\n")
    with Pool(args.num_workers) as p:
        results_iterator = p.imap_unordered(validate_sample, tasks)
        for result in tqdm(results_iterator, total=total_samples, desc="Processing samples"):
            # Print detailed debug for every sample
            if args.debug_print:
                try:
                    print_sample_debug(result)
                except Exception:
                    pass
            # Collect only passing samples
            if isinstance(result, dict) and result.get('ok'):
                passed_samples.append(result['sample'])
            elif isinstance(result, dict):
                r = result.get('reason', 'unknown')
                failure_counts[r] = failure_counts.get(r, 0) + 1
                if args.failed_output_dir:
                    try:
                        failed_entry: Dict[str, Any] = {
                            "sample_id": result.get('sample_id'),
                            "reason": r,
                            "sample": result.get('sample'),
                        }
                        for k in ("case_index", "input", "expected", "actual", "stderr", "stdout", "status_code", "text", "error"):
                            if k in result:
                                failed_entry[k] = result[k]
                        failed_samples.append(failed_entry)
                    except Exception:
                        pass

    print("\nValidation complete.")
    num_passed = len(passed_samples)
    success_rate = (num_passed / total_samples) * 100 if total_samples > 0 else 0

    print("\n--- Filtering Summary ---")
    print(f"Total samples processed: {total_samples}")
    print(f"Solutions that passed all public & private tests: {num_passed}")
    print(f"Success Rate: {success_rate:.2f}%")

    if failure_counts:
        print("\nFailure breakdown by reason:")
        for k in sorted(failure_counts.keys()):
            print(f"- {k}: {failure_counts[k]}")

    if num_passed > 0:
        print(f"\nSaving {num_passed} high-quality samples to '{args.output_path}' in Arrow format...")
        final_dataset = Dataset.from_list(passed_samples)
        try:
            final_dataset.save_to_disk(args.output_path)
            print(f"Successfully saved to {args.output_path}")
        except Exception as e:
            print(f"Error saving file: {e}")
    else:
        print("\nNo samples passed. Nothing to save.")

    if args.failed_output_dir and failed_samples:
        try:
            failed_rows: List[Dict[str, Any]] = []
            for row in failed_samples:
                base = row.get("sample") or {}
                if isinstance(base, dict):
                    enriched = dict(base)
                    enriched["fail_reason"] = row.get("reason")
                    if "case_index" in row: enriched["fail_case_index"] = row["case_index"]
                    if "input" in row: enriched["fail_input"] = row["input"]
                    if "expected" in row: enriched["fail_expected"] = row["expected"]
                    if "actual" in row: enriched["fail_actual"] = row["actual"]
                    if "stderr" in row: enriched["fail_stderr"] = row["stderr"]
                    if "stdout" in row: enriched["fail_stdout"] = row["stdout"]
                    if "status_code" in row: enriched["fail_status_code"] = row["status_code"]
                    if "text" in row: enriched["fail_text"] = row["text"]
                    if "error" in row: enriched["fail_error"] = row["error"]
                    failed_rows.append(enriched)
            if failed_rows:
                failed_ds = Dataset.from_list(failed_rows)
                failed_ds.save_to_disk(args.failed_output_dir)
                print(f"Saved {len(failed_rows)} failed samples to {args.failed_output_dir}")
            else:
                print("No failed rows to save.")
        except Exception as e:
            print(f"Error saving failed samples (arrow): {e}")

    print("Done. Your dataset filtering run is complete.")

if __name__ == "__main__":
    main()