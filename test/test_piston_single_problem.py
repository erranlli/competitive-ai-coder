#!/usr/bin/env python3
import argparse
import json
import os
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from datasets import load_dataset

DEFAULT_ENDPOINT = "http://localhost:2000"


def _normalize_text_basic(text: str) -> str:
    if text is None:
        return ""
    lines = text.splitlines()
    return "\n".join(line.rstrip() for line in lines).strip()


FLOAT_ABS_TOL = 1e-6
FLOAT_REL_TOL = 1e-9


def _is_number_token(s: str) -> bool:
    try:
        Decimal(s)
        return True
    except (InvalidOperation, TypeError):
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


def _compare_json_like(a_str: str, b_str: str) -> Optional[bool]:
    try:
        a = json.loads(a_str)
        b = json.loads(b_str)
    except Exception:
        return None

    def cmp(x: Any, y: Any) -> bool:
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
    expected_normalized = expected.replace('\\n', '\n')
    actual_normalized = actual.replace('\\n', '\n')
    e = _normalize_text_basic(expected_normalized)
    a = _normalize_text_basic(actual_normalized)
    if e == a:
        return True
    js_cmp = _compare_json_like(e, a)
    if js_cmp is True:
        return True
    if js_cmp is False:
        return False
    e_lines = [ln for ln in e.splitlines() if ln.strip() != ""]
    a_lines = [ln for ln in a.splitlines() if ln.strip() != ""]
    if e_lines == a_lines:
        return True
    return False


def checker_says_pass(stdout_text: str) -> Optional[bool]:
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


def _print_detail_block(title: str, content: str, max_chars: int = 10000) -> None:
    print(f"{title}:")
    text = content if content is not None else ""
    if len(text) > max_chars:
        print(text[:max_chars])
        print(f"... [truncated {len(text) - max_chars} chars]")
    else:
        print(text)


def run_piston_test(
    source_code: str,
    problem_data: Dict[str, Any],
    test_case: Dict[str, str],
    language: str = "python",
    extension: str = "py",
    endpoint: str = DEFAULT_ENDPOINT,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    input_text = test_case.get("input", "") or ""
    expected_output = (test_case.get("output", "") or "").strip()

    stdin_threshold_bytes_env = 0
    try:
        stdin_threshold_bytes = int(stdin_threshold_bytes_env)
    except Exception:
        stdin_threshold_bytes = 4_000_000
    use_stdin = len(input_text) < stdin_threshold_bytes

    wrapped_source_code = source_code
    if not use_stdin and language == "python" and extension == "py":
        prelude = (
            "import sys\n"
            "try:\n"
            "    sys.stdin = open('input.txt', 'r')\n"
            "except Exception:\n"
            "    pass\n\n"
        )
        wrapped_source_code = prelude + source_code

    files = [
        {"name": f"main.{extension}", "content": wrapped_source_code},
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
            return {"passed": False, "output": "", "expected": expected_output, "error": f"HTTP {result.status_code}: {result.text}", "reason": "http_error", "raw_response": result.text}

        response = result.json()
        compile_info = response.get("compile", {})
        run_info = response.get("run", {})

        compile_code = compile_info.get("code", 0)
        run_code = run_info.get("code", 1)
        stdout = run_info.get("stdout", "") or ""
        stderr = run_info.get("stderr", "") or ""

        if compile_code != 0:
            return {
                "passed": False,
                "output": stdout,
                "expected": expected_output,
                "error": compile_info.get("stderr", "") or "compile_error",
                "reason": "compile_error",
            }

        if run_code != 0:
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
                "expected": expected_output,
                "error": stderr or run_info.get("stderr", "") or "runtime_error",
                "reason": reason,
            }

        solution_output = run_info.get("stdout", "")
    except Exception as e:
        return {"passed": False, "output": "", "expected": expected_output, "error": str(e), "reason": "exception"}

    solution_output = (solution_output or "").strip()
    checker_code = problem_data.get("generated_checker")

    if not checker_code:
        ok = outputs_match(expected_output, solution_output)
        return {
            "passed": ok,
            "output": solution_output,
            "expected": expected_output,
            "error": None if ok else "outputs_mismatch",
            "reason": "passed_by_exact_match" if ok else "failed_outputs_match",
        }
    else:
        checker_payload = {
            "language": "python",
            "version": "*",
            "files": [
                {"name": "checker.py", "content": checker_code},
                {"name": "input.txt", "content": input_text},
                {"name": "correct_output.txt", "content": expected_output},
                {"name": "solution_output.txt", "content": solution_output},
            ],
            "args": ["input.txt", "correct_output.txt", "solution_output.txt"],
        }

        try:
            checker_result = http.post(f"{endpoint}/api/v2/execute", json=checker_payload)
            if checker_result.status_code != 200:
                return {"passed": False, "output": solution_output, "expected": expected_output, "error": f"Checker HTTP {checker_result.status_code}", "reason": "http_error_checker"}

            checker_response = checker_result.json().get("run", {})
            checker_stdout = checker_response.get("stdout", "")
            checker_stderr = checker_response.get("stderr", "")

            checker_pass = checker_says_pass(checker_stdout)

            if checker_pass is True:
                return {"passed": True, "output": solution_output, "expected": expected_output, "error": None, "reason": "passed_by_checker"}
            else:
                return {"passed": False, "output": solution_output, "expected": expected_output, "error": checker_stderr or checker_stdout or "checker_declared_fail", "reason": "failed_checker"}
        except Exception as e:
            return {"passed": False, "output": solution_output, "expected": expected_output, "error": str(e), "reason": "exception_running_checker"}


def load_generated_tests(problem_id: str, generated_tests_dir: str) -> List[Dict[str, str]]:
    contest_id = str(problem_id).split("/")[0]
    test_file = Path(generated_tests_dir) / f"test_cases_{contest_id}.parquet"
    if not test_file.exists():
        return []
    try:
        df = pd.read_parquet(test_file)
        problem_tests = df[df["problem_id"] == problem_id]
        tests: List[Dict[str, str]] = []
        for _, row in problem_tests.iterrows():
            tests.append({
                "input": row["input"],
                "output": row["output"],
                "test_case_i": int(row["test_i"]) if "test_i" in row else int(row.get("test_case_i", 0)),
            })
        tests.sort(key=lambda x: x["test_case_i"])  # type: ignore
        return tests
    except Exception:
        return []


def build_problem_index() -> Dict[str, Dict[str, Any]]:
    id_to_row: Dict[str, Dict[str, Any]] = {}
    for split_name in ["verifiable", "default"]:
        try:
            ds = load_dataset("open-r1/codeforces", split="test", name=split_name)
            for r in ds:
                rid = r.get("id") or (str(r.get("contest_id", "")) + "/" + str(r.get("index", "")))
                if rid and rid not in id_to_row:
                    id_to_row[rid] = r
        except Exception:
            continue
    return id_to_row


def pick_solution_code_from_jsonl(solutions_path: str, problem_id: str) -> Optional[str]:
    code: Optional[str] = None
    with open(solutions_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            except Exception:
                continue
            rid = item.get("id") or ((str(item.get("contest_id", "")) + "/" + str(item.get("index", ""))) if item.get("contest_id") else None)
            if rid == problem_id:
                #breakpoint()
                code = item.get("code") or item.get("solution") or None
                if code:
                    return code
    return code


def evaluate_single(problem_id: str, source_code: str, endpoint: str, generated_tests_dir: str, max_generated_tests: int) -> int:
    index = build_problem_index()
    prob = index.get(problem_id, {})
    if not prob:
        print(f"Problem {problem_id} not found in dataset")
        return 1

    official_tests: List[Dict[str, str]] = prob.get("official_tests") or []
    checker_present = bool(prob.get("generated_checker"))

    session = requests.Session()
    all_passed = True

    print("\n=== SOURCE CODE ===")
    _print_detail_block("Source Code", source_code)
    print(f"Checker present: {checker_present}")

    print("\n=== OFFICIAL TESTS ===")
    for i, case in enumerate(official_tests):
        res = run_piston_test(
            source_code,
            {
                "time_limit": prob.get("time_limit", 2.0),
                "memory_limit": prob.get("memory_limit", 256.0),
                "input_mode": prob.get("input_mode", "stdio"),
                "generated_checker": prob.get("generated_checker"),
            },
            case,
            language="python",
            extension="py",
            endpoint=endpoint,
            session=session,
        )
        passed = bool(res.get("passed", False))
        reason = res.get("reason")
        print(f"\nOfficial {i+1}: {'PASS' if passed else 'FAIL'} ({reason})")
        _print_detail_block("Input", case.get("input", ""))
        _print_detail_block("Expected", (case.get("output", "") or "").strip())
        _print_detail_block("Model Output", res.get("output", "") or "")
        if res.get("error"):
            _print_detail_block("Error", res.get("error", ""))
        if not passed:
            all_passed = False

    gen_tests = load_generated_tests(problem_id, generated_tests_dir)
    if gen_tests and max_generated_tests != 0:
        print("\n=== GENERATED TESTS ===")
        limited = gen_tests if max_generated_tests < 0 else gen_tests[:max_generated_tests]
        for case in limited:
            res = run_piston_test(
                source_code,
                {
                    "time_limit": prob.get("time_limit", 2.0),
                    "memory_limit": prob.get("memory_limit", 256.0),
                    "input_mode": prob.get("input_mode", "stdio"),
                    "generated_checker": prob.get("generated_checker"),
                },
                case,
                language="python",
                extension="py",
                endpoint=endpoint,
                session=session,
            )
            passed = bool(res.get("passed", False))
            reason = res.get("reason")
            print(f"\nGenerated {case.get('test_case_i')}: {'PASS' if passed else 'FAIL'} ({reason})")
            _print_detail_block("Input", case.get("input", ""))
            _print_detail_block("Expected", (case.get("output", "") or "").strip())
            _print_detail_block("Model Output", res.get("output", "") or "")
            if res.get("error"):
                _print_detail_block("Error", res.get("error", ""))
            if not passed:
                all_passed = False

    print("\n=== SUMMARY ===")
    if all_passed:
        print("All evaluated tests passed.")
        return 0
    else:
        print("Some tests failed.")
        return 2


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a single problem solution via Piston (Python only)")
    p.add_argument("--problem-id", required=True, help="Problem id like 2063/A")
    p.add_argument("--solutions-path", required=True, help="Path to model_solutions JSONL file")
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--generated-tests-dir", default="/root/takehome/open_rl_codeforces/generated_tests")
    p.add_argument("--max-generated-tests", type=int, default=0, help="0=disable, -1=all, N=first N")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    code = pick_solution_code_from_jsonl(args.solutions_path, args.problem_id)
    if not code:
        print(f"No code found for {args.problem_id} in {args.solutions_path}")
        raise SystemExit(1)
    exit_code = evaluate_single(args.problem_id, code, args.endpoint, args.generated_tests_dir, args.max_generated_tests)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()