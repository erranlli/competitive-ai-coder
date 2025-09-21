#!/usr/bin/env python3
"""
filter_and_save_cots.py (v3 - Corrected Test Parsing)

Validates solutions from the 'open-r1/codeforces-cots' dataset against ALL
verifiable tests (public and private). It correctly parses the test dictionary
structure and saves high-quality samples in Arrow format for training.

Usage:
    python filter_and_save_cots.py --endpoint http://localhost:2000 \
        --output-path ./codeforces_cots_high_quality.arrow \
        --num-workers 16
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
#  Do not train with this!
#  SUBSET_NAME = "solutions_w_editorials_py_decontaminated"
SUBSET_NAME = "solutions_py_decontaminated" #8133 samples


# --- Robust Output Matching Logic ---

def _normalize_text_basic(text: str) -> str:
    if text is None:
        return ""
    lines = text.splitlines()
    return "\n".join(line.rstrip() for line in lines).strip()

def outputs_match(expected: str, actual: str, abs_tol: float = 1e-6, rel_tol: float = 1e-6) -> bool:
    """Robust output matcher:
    - Exact match on normalized text passes
    - Otherwise, compare token-by-token ignoring whitespace layout
      - Numeric tokens: compare within tolerance (abs or relative)
      - Text tokens: case-insensitive exact match
    """
    e_raw = _normalize_text_basic(expected)
    a_raw = _normalize_text_basic(actual)
    if e_raw == a_raw:
        return True

    # Tokenize by whitespace
    e_tokens = e_raw.split()
    a_tokens = a_raw.split()
    if len(e_tokens) == len(a_tokens):
        # Try strict token-wise compare with tolerance
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

    # Looser matching on counts: compare as multisets ignoring ordering and line breaks
    from collections import Counter

    def _partition_tokens(tokens: list[str]) -> tuple[list[float], Counter]:
        nums: list[float] = []
        texts: Counter = Counter()
        for t in tokens:
            try:
                nums.append(float(t))
            except Exception:
                texts[t.lower()] += 1
        return nums, texts

    e_nums, e_texts = _partition_tokens(e_tokens)
    a_nums, a_texts = _partition_tokens(a_tokens)

    # Text tokens must match as a multiset (case-insensitive)
    if e_texts != a_texts:
        return False

    # Numeric tokens: greedy tolerance matching
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

def _is_special_sample(sample: dict, sample_id: str | None) -> bool:
    try:
        if sample_id in {"213/C", "1375/G", "424/D"}:
            return True
        cid = sample.get("contest_id")
        idx = sample.get("index") or sample.get("problem_index")
        if ((cid == 213 and str(idx).upper() == "C") or
            (cid == 1375 and str(idx).upper() == "G") or
            (cid == 424 and str(idx).upper() == "D")):
            return True
        # Also check aliases if present
        aliases = sample.get("aliases") or []
        if isinstance(aliases, list) and any(a in {"213/C", "1375/G", "424/D"} for a in aliases):
            return True
    except Exception:
        pass
    return False

def parse_and_combine_tests(sample: dict) -> Tuple[List[Tuple[str, str]], str]:
    """
    Parses public and private tests from the sample and combines them.
    The test format is a dict {'input': [in1, in2], 'output': [out1, out2]}.
    """
    all_tests: List[Tuple[str, str]] = []
    sources_used: set[str] = set()
    # We can only use tests that have both inputs and expected outputs
    #test_keys_to_check = ['public_tests', 'private_tests', 'generated_tests']
    test_keys_to_check = ['public_tests', 'private_tests'] #TODO: not very useful


    for key in test_keys_to_check:
        test_data = sample.get(key)
        if (isinstance(test_data, dict) and
                'input' in test_data and 'output' in test_data and
                test_data['input'] and test_data['output'] and
                len(test_data['input']) == len(test_data['output'])):
            
            inputs = test_data['input']
            outputs = test_data['output']
            # Create pairs of (input, output) for each test case
            all_tests.extend(zip(inputs, outputs))
            sources_used.add(key)

    # Fallback: if no public/private tests, try generated_tests
    if not all_tests:
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
            if not outputs_match(expected_output, actual_output):
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
    parser = argparse.ArgumentParser(description="Filter codeforces-cots dataset for high-quality training samples.")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="URL of your local Piston API server.")
    parser.add_argument("--output-path", default="/mnt/data2/new_codeforces_cots_high_quality_arrow", help="Path to save the filtered Arrow dataset.")
    parser.add_argument("--num-workers", type=int, default=cpu_count(), help="Number of parallel processes to use.") 
    #parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel processes to use.")  #TODO
    parser.add_argument("--failed-output-dir", default="/mnt/data2/new_codeforces_cots_failed_arrow", help="Directory to save failed samples (Arrow, save_to_disk)")

    args = parser.parse_args()

    print(f"Loading dataset '{DATASET_NAME}' subset '{SUBSET_NAME}'...")
    dataset = load_dataset(DATASET_NAME, SUBSET_NAME, split='train', streaming=True) # for debugging
    #dataset = load_dataset(DATASET_NAME, SUBSET_NAME, split='train[:300]') # for debugging

    all_samples = list(dataset)
    total_samples = len(all_samples)
    print(f"Loaded {total_samples} samples.")

    tasks = [(sample, args.endpoint) for sample in all_samples]
    print(f"\nStarting validation using {args.num_workers} parallel workers...")
    
    passed_samples = []
    failure_counts: dict[str, int] = {}
    failed_samples: list[dict[str, Any]] = []

    def print_sample_debug(result: Dict[str, Any]) -> None:
        MAX_CODE = 2000
        code = result.get('code') or ''
        code_print = code if len(code) <= MAX_CODE else code[:MAX_CODE] + f"... [truncated {len(code) - MAX_CODE} chars]"
        sid = result.get('sample_id', '<unknown>')
        if result.get('ok'):
            print(f"\n=== SAMPLE PASSED === id={sid} tests={result.get('num_tests')}\n--- CODE ---\n{code_print}\n")
            # Always print raw for specific ids
            try:
                if _is_special_sample(result.get('sample', {}), sid):
                    print("SPECIAL RAW SAMPLE:")
                    print(json.dumps(result.get('sample', {}), ensure_ascii=False, indent=2))
            except Exception:
                pass
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
                # Print raw sample for no_code_extracted cases
                if reason == 'no_code_extracted':
                    try:
                        print("RAW SAMPLE:")
                        print(json.dumps(result.get('sample', {}), ensure_ascii=False))
                    except Exception:
                        pass
            # Always print raw for specific ids
            try:
                if _is_special_sample(result.get('sample', {}), sid):
                    print("SPECIAL RAW SAMPLE:")
                    print(json.dumps(result.get('sample', {}), ensure_ascii=False, indent=2))
            except Exception:
                pass
            print(f"--- CODE ---\n{code_print}\n")
    with Pool(args.num_workers) as p:
        results_iterator = p.imap_unordered(validate_sample, tasks)
        for result in tqdm(results_iterator, total=total_samples, desc="Processing samples"):
            # Print detailed debug for every sample
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
                # Collect failed sample with metadata
                try:
                    failed_entry: Dict[str, Any] = {
                        "sample_id": result.get('sample_id'),
                        "reason": r,
                        "sample": result.get('sample'),
                    }
                    # Optional diagnostics
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

    # Failure breakdown
    if failure_counts:
        print("\nFailure breakdown by reason:")
        for k in sorted(failure_counts.keys()):
            print(f"- {k}: {failure_counts[k]}")

    if num_passed == 0:
        print("\nNo samples passed. Nothing to save.")
        # Still save failed samples as Arrow with same schema structure as source samples
        try:
            failed_rows: list[dict[str, Any]] = []
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
        return

    print(f"\nSaving {num_passed} high-quality samples to '{args.output_path}' in Arrow format...")
    final_dataset = Dataset.from_list(passed_samples)
    try:
        final_dataset.save_to_disk(args.output_path)
        print(f"Successfully saved to {args.output_path}")
    except Exception as e:
        print(f"Error saving file: {e}")

    # Save failed samples as Arrow for visualization parity
    try:
        failed_rows: list[dict[str, Any]] = []
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

    print("Done. Your new dataset is ready for training.")

if __name__ == "__main__":
    main()