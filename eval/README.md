# Evaluation with Piston (+ Generated Tests and Checkers)

Scripts to evaluate model solutions against the `open-r1/codeforces` dataset using a Piston-compatible execution service, with support for problem-provided checkers and generated tests. The main entrypoint is `eval_with_piston_gentest_checker_stats.py`.

## Important: Piston server defaults are not enough

The default Piston server configuration often fails for our workloads:
 - JSON body/`stdin` size limits can truncate big inputs
 - Inconsistent error reporting under large payloads

We work around this by replacing stdin with a file when inputs are large:
 - Embed large input as `input.txt` in Piston `files`
 - Prepend a Python prelude to the user code to redirect `sys.stdin` to `input.txt`
 - Keep JSON `stdin` empty in that case

This logic is implemented in `run_piston_test()` in `eval_with_piston_gentest_checker_stats.py` and is required for stable evaluation.

## Quickstart

Ensure a Piston-compatible API is reachable (defaults to `http://localhost:2000`). Then run:

```bash
python eval/eval_with_piston_gentest_checker_stats.py \
  --solutions-path model_solutions/your_model__open-r1-codeforces__default__test__vllm.jsonl \
  --endpoint http://localhost:2000 \
  --generated-tests-dir /path/to/generated_tests \
  --max-generated-tests 0 \
  --results-dir results
```

Key options:
 - `--max-generated-tests`: 0 disables, -1 all, N first N
 - `--sort-generated-tests`: `none | small_first | large_first`
 - `--skip-large-inputs-over`: skip excessively large inputs (bytes)
 - `--max-generated-bytes-per-problem`: cap generated input bytes budget
 - `--generated-tests-sample`: (0,1] downsample fraction
 - `--generated-tests-workers`: >1 enables concurrent generated tests

## Server setup (example)

If you deploy your own server, raise payload/time/memory limits. Example approach with Piston:

```bash
git clone https://github.com/engineer-man/piston
cd piston
docker build -t piston-fixed .
docker stop piston_api || true && docker rm piston_api || true
docker run --privileged -dit -p 2000:2000 --name piston_api --env-file ./piston.env piston-fixed
cli/index.js ppman install python
```

Even with relaxed limits, keep the stdin-as-file workaround in place for robustness.

## Outputs

- Results JSONL: `results/<solutions_prefix>/piston_eval_results.jsonl`
- Metrics JSON: `results/<solutions_prefix>/piston_eval_metrics.json`

Per-case records include: `case_type` (official/generated), `test_case_i` (if available), `passed`, `output`, `expected`, `error`, `reason`.

## Dependencies

- `datasets`, `pandas`, `requests`
- Python 3.10+

## Troubleshooting

- Connection errors: verify endpoint/container health
- Runtime errors with big inputs: confirm stdin replacement occurred; try `--sort-generated-tests small_first`
- Checker misreporting: some checkers are ambiguous; see `failed_checker`/`checker_unclear` and inspect `stdout`/`stderr`