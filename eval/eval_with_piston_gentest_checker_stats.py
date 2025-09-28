#!/usr/bin/env python3
"""
piston_eval_with_stats.py

Usage:
    python piston_eval_with_stats.py --solutions-path /path/to/solutions.jsonl \
        --endpoint http://localhost:2000 \
        --generated-tests-dir /path/to/generated_tests \
        --max-generated-tests -1 \
        --results-dir /path/to/results
"""

import argparse
import json
import os
import time
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from datasets import load_dataset

DEFAULT_ENDPOINT = "http://localhost:2000"

# ----------------------------
# Utility functions
# ----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# Basic normalization
def _normalize_text_basic(text: str) -> str:
    """A robust text normalizer."""
    if text is None:
        return ""
    
    # .splitlines() handles all line endings (\n, \r, \r\n) automatically.
    lines = text.splitlines()
    
    # Re-join the lines after stripping trailing whitespace from each one,
    # then strip any leading/trailing blank lines from the entire block.
    return "\n".join(line.rstrip() for line in lines).strip()


# ----------------------------
# Smart outputs_match (fallback)
# ----------------------------
FLOAT_ABS_TOL = 1e-6
FLOAT_REL_TOL = 1e-9

def _is_number_token(s: str) -> bool:
    try:
        Decimal(s)
        return True
    except (InvalidOperation, TypeError):
        # try float fallback
        try:
            float(s)
            return True
        except Exception:
            return False

def _to_number(s: str) -> Decimal:
    try:
        return Decimal(s)
    except InvalidOperation:
        return Decimal(str(float(s)))

def _numbers_equal(a: Decimal, b: Decimal, abs_tol: float = FLOAT_ABS_TOL, rel_tol: float = FLOAT_REL_TOL) -> bool:
    diff = abs(a - b)
    if diff <= Decimal(str(abs_tol)):
        return True
    maxab = max(abs(a), abs(b))
    if maxab == 0:
        return diff == 0
    return (diff / maxab) <= Decimal(str(rel_tol))

def _compare_number_multisets(text1: str, text2: str) -> bool:
    def tokens(s: str) -> List[Decimal]:
        parts = s.replace(",", " ").split()
        nums = []
        for p in parts:
            if _is_number_token(p):
                nums.append(_to_number(p))
        return nums

    n1 = tokens(text1)
    n2 = tokens(text2)
    if not n1 and not n2:
        return False
    if len(n1) != len(n2):
        return False

    used = [False] * len(n2)
    for a in n1:
        matched = False
        for i, b in enumerate(n2):
            if not used[i] and _numbers_equal(a, b):
                used[i] = True
                matched = True
                break
        if not matched:
            return False
    return True

def _compare_json_like(a_str: str, b_str: str) -> Optional[bool]:
    try:
        a = json.loads(a_str)
        b = json.loads(b_str)
    except Exception:
        return None

    def cmp(x: Any, y: Any) -> bool:
        # numeric cross-type
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return _numbers_equal(Decimal(str(x)), Decimal(str(y)))
        if type(x) != type(y):
            return False
        if isinstance(x, dict):
            if set(x.keys()) != set(y.keys()):
                return False
            return all(cmp(x[k], y[k]) for k in x.keys())
        if isinstance(x, list):
            if len(x) != len(y):
                return False
            return all(cmp(xi, yi) for xi, yi in zip(x, y))
        if isinstance(x, (int, float)):
            return _numbers_equal(Decimal(str(x)), Decimal(str(y)))
        return x == y

    return cmp(a, b)


def outputs_match(expected: str, actual: str) -> bool:
    # Normalize newlines for comparison 
    expected_normalized = expected.replace('\\n', '\n')
    actual_normalized = actual.replace('\\n', '\n')
    e = _normalize_text_basic(expected_normalized)
    a = _normalize_text_basic(actual_normalized)

    # Fast exact path
    if e == a:
        return True

    # JSON-aware
    js_cmp = _compare_json_like(e, a)
    if js_cmp is True:
        return True
    if js_cmp is False:
        return False

    # Numeric multiset compare (order-invariant numeric lists)
    #if _compare_number_multisets(e, a):
    #    return True

    # Fallback: compare non-empty line sequences
    e_lines = [ln for ln in e.splitlines() if ln.strip() != ""]
    a_lines = [ln for ln in a.splitlines() if ln.strip() != ""]
    if e_lines == a_lines:
        return True

    return False

# ----------------------------
# Checker output parsing helper
# ----------------------------
def checker_says_pass(stdout_text: str) -> Optional[bool]:
    toks = stdout_text.strip().split()
    if not toks:
        return None
    first = toks[0]
    try:
        val = float(first)
        return val > 0
    except Exception:
        if first.startswith("1"):
            return True
        if first.startswith("0"):
            return False
        return None

# ----------------------------
# Run a single test via Piston (checker-aware)
# ----------------------------
import os
import requests
from typing import Dict, Any, Optional

# Assume these helper functions and constants are defined elsewhere
DEFAULT_ENDPOINT = "http://localhost:2000"


def checker_says_pass(stdout_text: str) -> Optional[bool]:
    """Parses the output of a checker program to determine pass/fail."""
    toks = stdout_text.strip().split()
    if not toks:
        return None
    first = toks[0]
    try:
        val = float(first)
        return val > 0
    except Exception:
        if first.upper().startswith("OK") or first.upper().startswith("AC"):
            return True
        if first.upper().startswith("WA") or first.upper().startswith("FAIL"):
            return False
        if first.startswith("1"):
            return True
        if first.startswith("0"):
            return False
        return None


# ----------------------------
# Run a single test via Piston (checker-aware)
# ----------------------------

def run_piston_test(
    source_code: str,
    problem_data: Dict[str, Any],
    test_case: Dict[str, str],
    language: str = "python",
    extension: str = "py",
    endpoint: str = DEFAULT_ENDPOINT,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    # --- Part 1: Prepare and Run the User's Solution ---

    # Avoid sending extremely large data via JSON stdin to Piston (can be truncated by server limits).
    # For large inputs, embed as file and wrap the user's program to read from input.txt instead of stdin.
    input_text = test_case.get("input", "") or ""
    output_text = test_case.get("output", "") or ""

    # Threshold can be tuned via env; default ~4MB
    stdin_threshold_bytes_env = 0 # TODO os.environ.get("PISTON_STDIN_THRESHOLD_BYTES", "4000000")
    try:
        stdin_threshold_bytes = int(stdin_threshold_bytes_env)
    except Exception:
        stdin_threshold_bytes = 4_000_000

    use_stdin = len(input_text) < stdin_threshold_bytes

    # Optionally wrap the user's code to read from input.txt when not using stdin
    wrapped_source_code = source_code
    if not use_stdin and language == "python" and extension == "py":
        prelude = (
            "import sys\n"
            "try:\n"
            "    # Replace stdin with file to avoid API stdin size limits\n"
            "    sys.stdin = open('input.txt', 'r')\n"
            "except Exception:\n"
            "    pass\n\n"
        )
        wrapped_source_code = prelude + source_code

    files = [
        {"name": f"main.{extension}", "content": wrapped_source_code},
        # Only send input.txt when not using stdin to keep payload small
        *([{ "name": "input.txt", "content": input_text }] if not use_stdin else []),
        {
            "name": "grader_config",
            "content": "\n".join(
                f"{key}={value}" for key, value in {
                    "TIME_LIMIT": problem_data.get("time_limit", 2.0),
                    "MEMORY_LIMIT": problem_data.get("memory_limit", 256.0),
                    "INPUT_MODE": problem_data.get("input_mode", "stdio"),
                }.items()
            ),
        },
    ]
    # Remove potential None entries (when output_text empty)
    files = [f for f in files if f is not None]

    payload = {
        "language": language,
        "version": "*",
        "files": files,
        "stdin": input_text if use_stdin else "",
        "run_timeout": int(float(problem_data.get("time_limit", 2.0)) * 1000),
        "run_memory_limit": int(float(problem_data.get("memory_limit", 256.0)) * 1024 * 1024),
    }

    try:
        http = session or requests
        result = http.post(f"{endpoint}/api/v2/execute", json=payload, headers={"Content-Type": "application/json"})
        if result.status_code != 200:
            return {"passed": False, "output": "", "expected": test_case.get("output", ""), "error": f"HTTP {result.status_code}: {result.text}", "reason": "http_error", "raw_response": result.text}

        response = result.json()
        compile_info = response.get("compile", {})
        run_info = response.get("run", {})

        compile_code = compile_info.get("code", 0)  # many langs set 0 even when skip-compile
        run_code = run_info.get("code", 1)
        stdout = run_info.get("stdout", "") or ""
        stderr = run_info.get("stderr", "") or ""

        # classify low-level failures
        if compile_code != 0:
            return {
                "passed": False,
                "output": stdout,
                "expected": test_case.get("output", ""),
                "error": compile_info.get("stderr", "") or "compile_error",
                "reason": "compile_error",
            }

        # run-level non-zero code -> runtime/TLE/MLE/RE
        if run_code != 0:
            # heuristics on stderr or stdout
            low = (stderr + "\n" + stdout).lower()
            if "time limit" in low or "took too long" in low or "timeout" in low or "tlim" in low or "t e" in low:
                reason = "time_limit_exceeded"
            elif "memory" in low or "memory limit" in low or "m e" in low:
                reason = "memory_limit_exceeded"
            else:
                reason = "runtime_error"
            return {
                "passed": False,
                "output": stdout,
                "expected": test_case.get("output", ""),
                "error": stderr or run_info.get("stderr", "") or "runtime_error",
                "reason": reason,
            }

        solution_output = run_info.get("stdout", "")
    except Exception as e:
        return {"passed": False, "output": "", "expected": test_case.get("output", ""), "error": str(e), "reason": "exception"}

    # --- Part 2: Judge the Output (either with a checker or by exact match) ---
    solution_output = run_info.get("stdout", "").strip()
    checker_code = problem_data.get("generated_checker")
    expected_output = test_case.get("output", "").strip() # Also strip expected output for consistency

    if not checker_code:
        # No checker -> fallback to simple exact match
        ok = outputs_match(expected_output, solution_output)
        return {
            "passed": ok, "output": solution_output, "expected": expected_output,
            "error": None if ok else "outputs_mismatch",
            "reason": "passed_by_exact_match" if ok else "failed_outputs_match",
        }
    else:
        # Checker exists -> run it in a second Piston call
        checker_payload = {
            "language": "python", # Checkers are python
            "version": "*",
            "files": [
                {"name": "checker.py", "content": checker_code},
                {"name": "input.txt", "content": input_text},
                {"name": "correct_output.txt", "content": expected_output},
                {"name": "solution_output.txt", "content": solution_output},
            ],
            "args": ["input.txt", "correct_output.txt", "solution_output.txt"] #SHIT: do not pass "checker.py" as an argument
        }

        #test_index = test_case.get('test_case_i', 'N/A')
        #import sys
        #print(f"\n--- DEBUGGING SANITY CHECK FOR TEST CASE INDEX: {test_index} ---", file=sys.stderr)
        #print(f"Checker Code (starts with): {checker_code[:100]}", file=sys.stderr)
        #print(f"Input Text (starts with):   {input_text[:100]}", file=sys.stderr)
        #print("-----------------------------------------------------------\n", file=sys.stderr)
    # --- END OF MODIFIED DEBUGGING BLOCK ---

        try:
            checker_result = http.post(f"{endpoint}/api/v2/execute", json=checker_payload)
            if checker_result.status_code != 200:
                return {"passed": False, "output": solution_output, "error": f"Checker HTTP {checker_result.status_code}", "reason": "http_error_checker"}
            
            checker_response = checker_result.json().get("run", {})
            checker_stdout = checker_response.get("stdout", "")
            checker_stderr = checker_response.get("stderr", "")

            checker_pass = checker_says_pass(checker_stdout)

            if checker_pass is True:
                return {"passed": True, "output": solution_output, "expected": expected_output, "error": None, "reason": "passed_by_checker"}
            else: # False or None (ambiguous) are both treated as failure
                return {"passed": False, "output": solution_output, "expected": expected_output, "error": checker_stderr or checker_stdout or "checker_declared_fail", "reason": "failed_checker"}

        except Exception as e:
            return {"passed": False, "output": solution_output, "error": str(e), "reason": "exception_running_checker"}



# ----------------------------
# Load generated tests
# ----------------------------
def load_generated_tests(problem_id: str, generated_tests_dir: str) -> List[Dict[str, str]]:
    from pathlib import Path

    contest_id = str(problem_id).split("/")[0]
    test_file = Path(generated_tests_dir) / f"test_cases_{contest_id}.parquet"
    if not test_file.exists():
        return []
    try:
        df = pd.read_parquet(test_file)
        problem_tests = df[df["problem_id"] == problem_id]
        tests: List[Dict[str, str]] = []
        for _, row in problem_tests.iterrows():
            tests.append({"input": row["input"], "output": row["output"], "test_case_i": int(row["test_i"])}) # Note: the dataset does not use test_case_i
        tests.sort(key=lambda x: x["test_case_i"])  # type: ignore
        return tests
    except Exception:
        return []

# ----------------------------
# Evaluate solutions file and gather stats
# ----------------------------
def evaluate_records(
    solutions_path: str,
    endpoint: str,
    generated_tests_dir: str,
    max_generated_tests: int = 10,
    results_dir: str = "/root/competitive-coding-ai/results",
    stop_on_first_failure: bool = False,
    skip_large_inputs_over: int = -1,
    sort_generated_tests: str = "none",  # none | small_first | large_first
    max_generated_bytes_per_problem: int = -1,
    generated_tests_workers: int = 1,
    generated_tests_sample: float = 1.0,
) -> str:
    # Derive subdirectory from solutions file prefix (basename without extension)
    solutions_prefix = os.path.splitext(os.path.basename(solutions_path))[0]
    final_results_dir = os.path.join(results_dir, solutions_prefix)
    ensure_dir(final_results_dir)
    out_path = os.path.join(final_results_dir, "piston_eval_results.jsonl")
    with open(out_path, "w", encoding="utf-8"):
        pass

    # Build fast lookup index for problem metadata
    id_to_row: Dict[str, Dict[str, Any]] = {}
    for split_name in ["verifiable", "default"]:
        try:
            ds = load_dataset("open-r1/codeforces", split="test", name=split_name)
            for r in ds:
                rid = r.get("id") or (r.get("contest_id", "") + "/" + r.get("index", ""))
                if rid and rid not in id_to_row:
                    id_to_row[rid] = r
        except Exception:
            continue

    def fetch_problem(problem_id: str) -> Dict[str, Any]:
        return id_to_row.get(problem_id, {})

    total = 0
    correct = 0
    started = time.time()
    session = requests.Session()

    # Stats counters
    stats = {
        "num_problems_total": 0,
        "num_with_checker": 0,
        "with_checker_passed": 0,
        "with_checker_failed": 0,
        "num_without_checker": 0,
        "without_checker_passed": 0,
        "failure_reasons": {},  # reason -> count
    }

    def incr_reason(reason: str):
        stats["failure_reasons"][reason] = stats["failure_reasons"].get(reason, 0) + 1

    with open(solutions_path, "r", encoding="utf-8") as f, open(out_path, "a", encoding="utf-8") as w:
        for line in f:
            try:
                item = json.loads(line)
            except Exception:
                continue

            problem_id = item.get("id") or ((item.get("contest_id", "") + "/" + item.get("index", "")) if item.get("contest_id") else None)
            if not problem_id:
                continue

            stats["num_problems_total"] += 1

            prob = fetch_problem(problem_id)
            official_tests: List[Dict[str, str]] = prob.get("official_tests") or []
            all_passed = True
            per_case: List[Dict[str, Any]] = []

            checker_present = bool(prob.get("generated_checker"))
            if checker_present:
                stats["num_with_checker"] += 1
            else:
                stats["num_without_checker"] += 1

            # If problem has a generated checker, we will use it for each test
            # Official tests
            for case in official_tests:
                res = run_piston_test(
                    item.get("code", ""),
                    {"time_limit": prob.get("time_limit", 2.0),
                     "memory_limit": prob.get("memory_limit", 256.0),
                     "input_mode": prob.get("input_mode", "stdio"),
                     "generated_checker": prob.get("generated_checker")},
                    case,
                    language="python",
                    extension="py",
                    endpoint=endpoint,
                    session=session
                )
                case_passed = res.get("passed", False)
                per_case.append({
                    "case_type": "official",
                    "passed": case_passed,
                    "output": res.get("output", ""),
                    "expected": case.get("output", ""),
                    "error": res.get("error", ""),
                    "reason": res.get("reason"),
                })
                if not case_passed:
                    all_passed = False
                    incr_reason(res.get("reason", "unknown"))

            # Generated tests
            gen_tests: List[Dict[str, str]] = None #load_generated_tests(problem_id, generated_tests_dir)
            if gen_tests:
                # Optional sampling
                if 0.0 < generated_tests_sample < 1.0:
                    # Keep order stable after sampling by sorting by test_case_i
                    sampled = []
                    for case in gen_tests:
                        if random.random() <= generated_tests_sample:
                            sampled.append(case)
                    gen_tests = sampled if sampled else gen_tests  # ensure not empty due to unlucky sampling

                # Sort by input size if requested
                if sort_generated_tests == "small_first":
                    gen_tests = sorted(gen_tests, key=lambda c: len(c.get("input", "")))
                elif sort_generated_tests == "large_first":
                    gen_tests = sorted(gen_tests, key=lambda c: len(c.get("input", "")), reverse=True)

                # Slice by max_generated_tests semantics: 0->none, -1->all, N>0->first N
                if max_generated_tests == 0:
                    limited_tests: List[Dict[str, str]] = []
                elif max_generated_tests > 0:
                    limited_tests = gen_tests[: max_generated_tests]
                else:
                    limited_tests = gen_tests

                # Apply per-problem bytes budget and per-case size skipping in a sequential or concurrent run
                total_bytes_used = 0

                def should_skip_case(case: Dict[str, str]) -> Optional[Dict[str, Any]]:
                    nonlocal total_bytes_used
                    case_input = case.get("input", "") or ""
                    case_size = len(case_input)
                    # Skip overly large inputs
                    if skip_large_inputs_over > 0 and case_size > skip_large_inputs_over:
                        return {
                            "case_type": "generated",
                            "test_case_i": case.get("test_case_i"),
                            "passed": False,
                            "output": "",
                            "expected": case.get("output", ""),
                            "error": f"skipped_large_input_over_{skip_large_inputs_over}",
                            "reason": "skipped_large_input",
                        }
                    # Enforce bytes budget
                    if max_generated_bytes_per_problem > 0 and (total_bytes_used + case_size) > max_generated_bytes_per_problem:
                        return {
                            "case_type": "generated",
                            "test_case_i": case.get("test_case_i"),
                            "passed": False,
                            "output": "",
                            "expected": case.get("output", ""),
                            "error": f"exceeded_generated_bytes_budget_{max_generated_bytes_per_problem}",
                            "reason": "budget_exceeded",
                        }
                    total_bytes_used += case_size
                    return None

                # Sequential mode with early stop
                if generated_tests_workers <= 1:
                    for case in limited_tests:
                        if stop_on_first_failure and not all_passed:
                            break
                        skip_record = should_skip_case(case)
                        if skip_record is not None:
                            per_case.append(skip_record)
                            all_passed = False
                            incr_reason(skip_record["reason"])  # type: ignore
                            if stop_on_first_failure:
                                break
                            continue
                        res = run_piston_test(
                            item.get("code", ""),
                            {"time_limit": prob.get("time_limit", 2.0),
                             "memory_limit": prob.get("memory_limit", 256.0),
                             "input_mode": prob.get("input_mode", "stdio"),
                             "generated_checker": prob.get("generated_checker")},
                            case,
                            language="python",
                            extension="py",
                            endpoint=endpoint,
                            session=session
                        )
                        case_passed = res.get("passed", False)
                        per_case.append({
                            "case_type": "generated",
                            "test_case_i": case.get("test_case_i"),
                            "passed": case_passed,
                            "output": res.get("output", ""),
                            "expected": case.get("output", ""),
                            "error": res.get("error", ""),
                            "reason": res.get("reason"),
                        })
                        if not case_passed:
                            all_passed = False
                            incr_reason(res.get("reason", "unknown"))
                else:
                    # Concurrent mode: submit tasks; early-stop not supported here (still benefits from small_first)
                    futures = []
                    with ThreadPoolExecutor(max_workers=generated_tests_workers) as ex:
                        for case in limited_tests:
                            skip_record = should_skip_case(case)
                            if skip_record is not None:
                                per_case.append(skip_record)
                                all_passed = False
                                incr_reason(skip_record["reason"])  # type: ignore
                                continue
                            futures.append((case, ex.submit(
                                run_piston_test,
                                item.get("code", ""),
                                {"time_limit": prob.get("time_limit", 2.0),
                                 "memory_limit": prob.get("memory_limit", 256.0),
                                 "input_mode": prob.get("input_mode", "stdio"),
                                 "generated_checker": prob.get("generated_checker")},
                                case,
                                "python",
                                "py",
                                endpoint,
                                None  # avoid sharing Session across threads
                            )))
                        for case, fut in futures:
                            res = fut.result()
                            case_passed = res.get("passed", False)
                            per_case.append({
                                "case_type": "generated",
                                "test_case_i": case.get("test_case_i"),
                                "passed": case_passed,
                                "output": res.get("output", ""),
                                "expected": case.get("output", ""),
                                "error": res.get("error", ""),
                                "reason": res.get("reason"),
                            })
                            if not case_passed:
                                all_passed = False
                                incr_reason(res.get("reason", "unknown"))

            # Update summary stats per problem (use official_tests as gating like before)
            total += 1
            # Count as correct if passes all official tests (when available) 
            # AND all generated tests (when available)
            # has_tests = len(official_tests) > 0 or len(gen_tests) > 0
            has_tests = (len(official_tests or []) > 0) or (len(gen_tests or []) > 0)
            if all_passed and has_tests:
                correct += 1
                #print(f"""***problem:{problem_id} Correct, passed all tests""")
                #for case in per_case:
                #    print(f"""code:{item.get('code', '')}""")
                #    print(f"""model output:{case.get('output', '')}, expected output:{case.get('expected', '')}""")
                #    print(f"""problem:{problem_id}, test_case:{case.get('test_case_i', '')}, error:{case.get('error','')}, reason:{case.get('reason', '')}""")

            #else:
            #    for case in per_case: #Erran: TODO
            #        if not case['passed']:
            #            print(f"""code:{item.get('code', '')}""")
            #            print(f"""model output:{case.get('output', '')}, expected output:{case.get('expected', '')}""")
            #            print(f"""problem:{problem_id}, test_case:{case.get('test_case_i', '')}, error:{case.get('error','')}, reason:{case.get('reason', '')}""")

            # For problem-level checker stats: if checker exists, use per-case reasons to decide pass/fail counts
            if checker_present:
                # treat a problem as passed_by_checker if all official tests passed and for generated tests either passed or no generated tests
                # But more granularly: if any per_case has reason "failed_checker" -> failed
                failed_checker_seen = any((pc.get("reason") == "failed_checker" or pc.get("reason") == "checker_unclear") for pc in per_case)
                if failed_checker_seen:
                    stats["with_checker_failed"] += 1
                else:
                    # If all testcases with checker either passed or were matched, count as passed
                    stats["with_checker_passed"] += 1
            else:
                # no checker: count as passed if all official tests passed (and existed)
                if all_passed and official_tests:
                    stats["without_checker_passed"] += 1

            record = {
                "problem_id": problem_id,
                "status": "passed" if (all_passed and official_tests) else "failed",
                "num_official": len(official_tests),
                "details": per_case,
            }
            w.write(json.dumps(record) + "\n")

            if total % 20 == 0:
                elapsed = time.time() - started
                avg = elapsed / max(1, total)
                eta = avg * max(0, (len(id_to_row) - total))
                print(f"Processed {total} problems | Correct: {correct} | pass@1: {correct / max(1, total):.3f} | avg {avg:.2f}s/problem | ETA ~{eta/60:.1f}m", flush=True)

    pass_at_1 = (correct / total) if total else 0.0
    metrics = {"num_attempted": total, "num_correct": correct, "pass_at_1": pass_at_1, "timestamp": time.time(), "stats": stats}
    metrics_path = os.path.join(final_results_dir, "piston_eval_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Print a human readable summary
    print("\n=== EVALUATION SUMMARY ===")
    print(json.dumps(metrics, indent=2))
    print("\nFailure reasons summary:")
    for reason, count in sorted(stats["failure_reasons"].items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")
    print(f"\nSaved detailed results to: {out_path}")
    print(f"Saved metrics to: {metrics_path}")
    return out_path

# ----------------------------
# Arg parser and main
# ----------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate generated solutions with Piston (checker-aware) and produce stats")
    p.add_argument("--solutions-path", default="/root/competitive-coding-ai/qwen-qwen2.5-7b-instruct__open-r1-codeforces__default__test__vllm_origin_t01_top095.jsonl")
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--generated-tests-dir", default="/root/takehome/open_rl_codeforces/generated_tests")
    p.add_argument("--max-generated-tests", type=int, default=0) # 0 disables, -1 all, N limits; TODO: all gen tests
    p.add_argument("--stop-on-first-failure", action="store_true")
    p.add_argument("--skip-large-inputs-over", type=int, default=50000000) # bytes #TODO: 50mb now
    p.add_argument("--sort-generated-tests", choices=["none", "small_first", "large_first"], default="none") #TODO: none
    p.add_argument("--max-generated-bytes-per-problem", type=int, default=-1)
    p.add_argument("--generated-tests-workers", type=int, default=1)
    p.add_argument("--generated-tests-sample", type=float, default=1.0) #TODO: no-sampling 
    p.add_argument("--results-dir", default="./results")
    return p

def main() -> None:
    args = build_arg_parser().parse_args()
    evaluate_records(
        args.solutions_path,
        args.endpoint,
        args.generated_tests_dir,
        args.max_generated_tests,
        args.results_dir,
        stop_on_first_failure=args.stop_on_first_failure,
        skip_large_inputs_over=args.skip_large_inputs_over,
        sort_generated_tests=args.sort_generated_tests,
        max_generated_bytes_per_problem=args.max_generated_bytes_per_problem,
        generated_tests_workers=args.generated_tests_workers,
        generated_tests_sample=args.generated_tests_sample,
    )

if __name__ == "__main__":
    main()
