## test: Piston Execution Smoke Tests

Small scripts to quickly verify code execution against Codeforces-style problems via a Piston-compatible API. Useful for sanity checks, runner configuration, and manual spot evaluation.

### Files

- `test_piston_running_ex.py`
  - Minimal end-to-end example that posts a Python solution and a single official test to the Piston endpoint.
  - Prints compile/run diagnostics and a basic output comparison (no custom checker).

- `test_piston_single_problem.py`
  - CLI tool to evaluate one problem end-to-end.
  - Loads the problem description and official tests from `open-r1/codeforces` (test split).
  - Extracts model solution code from a JSONL file and runs it against official and (optionally) generated tests.
  - Performs tolerant output comparison (including numeric tolerance and JSON-like structure when relevant).

### Requirements

- A Piston-compatible execution service accessible at `http://localhost:2000` (default).
  - If using Piston, ensure the API container is up (e.g., `docker-compose up -d api`).
- Python packages: `requests`, `datasets`, `pandas` (for generated test parquet loading), `ipython` optional.

### Quickstart

Run the simple example:

```bash
python test/test_piston_running_ex.py
```

Evaluate a specific problem with your model solutions JSONL:

```bash
python test/test_piston_single_problem.py \
  --problem-id 2063/C \
  --solutions-path /path/to/model_solutions/your_model__open-r1-codeforces__default__test__vllm.jsonl \
  --endpoint http://localhost:2000 \
  --generated-tests-dir /path/to/generated_tests \
  --max-generated-tests 0
```

Notes:
- `--max-generated-tests`: 0 disables, -1 uses all generated tests, N takes the first N.
- `--generated-tests-dir` expects per-contest parquet files like `test_cases_<contestId>.parquet` with columns `[problem_id, input, output, test_i]`.

### Troubleshooting

- Connection errors
  - Ensure the execution service is reachable at `--endpoint`.

- Runtime or compile errors
  - See `stderr` in the printed diagnostics; many problems require reading from stdin exactly as specified.

- Output mismatches for problems with multiple valid answers
  - Provide a custom checker via the dataset (supported by the runner) or rely on tolerant comparison and generated tests where appropriate.


