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
SUBSET_NAME = "solutions_w_editorials_py_decontaminated"

# --- Robust Output Matching Logic ---

def _normalize_text_basic(text: str) -> str:
    if text is None: return ""
    lines = text.splitlines()
    return "\n".join(line.rstrip() for line in lines).strip()

def outputs_match(expected: str, actual: str) -> bool:
    e = _normalize_text_basic(expected)
    a = _normalize_text_basic(actual)
    return e == a

# --- Helper Functions ---

def extract_python_code(generation_str: str) -> Optional[str]:
    """Extracts Python code from a markdown block."""
    match = re.search(r"```python\n(.*?)```", generation_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def parse_and_combine_tests(sample: dict) -> List[Tuple[str, str]]:
    """
    Parses public and private tests from the sample and combines them.
    The test format is a dict {'input': [in1, in2], 'output': [out1, out2]}.
    """
    all_tests = []
    # We can only use tests that have both inputs and expected outputs
    test_keys_to_check = ['public_tests', 'private_tests']

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
            
    return all_tests


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

    all_test_cases = parse_and_combine_tests(sample)
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
                    "reason": "outputs_mismatch",
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
    parser.add_argument("--output-path", default="./codeforces_cots_high_quality_arrow", help="Path to save the filtered Arrow dataset.")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel processes to use.") #cpu_count() TODO
    args = parser.parse_args()

    print(f"Loading dataset '{DATASET_NAME}' subset '{SUBSET_NAME}'...")
    #dataset = load_dataset(DATASET_NAME, SUBSET_NAME, split='train', streaming=False) #TODO
    dataset = load_dataset(DATASET_NAME, SUBSET_NAME, split='train[:20]')

    all_samples = list(dataset)
    total_samples = len(all_samples)
    print(f"Loaded {total_samples} samples.")

    tasks = [(sample, args.endpoint) for sample in all_samples]
    print(f"\nStarting validation using {args.num_workers} parallel workers...")
    
    passed_samples = []

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
            elif reason in ('no_code_extracted', 'no_valid_tests', 'exception'):
                extra = {k: v for k, v in result.items() if k not in {'ok','sample','code','sample_id','reason'}}
                if extra:
                    print(json.dumps(extra, ensure_ascii=False))
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

    print("\nValidation complete.")
    num_passed = len(passed_samples)
    success_rate = (num_passed / total_samples) * 100 if total_samples > 0 else 0

    print("\n--- Filtering Summary ---")
    print(f"Total samples processed: {total_samples}")
    print(f"Solutions that passed all public & private tests: {num_passed}")
    print(f"Success Rate: {success_rate:.2f}%")

    if num_passed == 0:
        print("\nNo samples passed. Nothing to save.")
        return

    print(f"\nSaving {num_passed} high-quality samples to '{args.output_path}' in Arrow format...")
    final_dataset = Dataset.from_list(passed_samples)
    try:
        final_dataset.save_to_disk(args.output_path)
        print(f"Successfully saved to {args.output_path}")
    except Exception as e:
        print(f"Error saving file: {e}")

    print("Done. Your new dataset is ready for training.")

if __name__ == "__main__":
    main()