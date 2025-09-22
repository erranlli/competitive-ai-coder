## datasets: Preparation, Conversion, and Visualization

Scripts and notebooks to prepare training/eval datasets, convert trajectories to CoT-style records, filter by test pass criteria, and visualize examples.

### Layout

- `mixture_of_thought.ipynb`: Large exploration notebook for MoT-style datasets; uses utilities from `data_util/` for pretty-printing and inspection.

- `preprocess_data/`
  - `filter_and_save_cots.py`: Consolidated filtering script.
    - Validates samples against tests via a Piston-like exec API and saves passing samples to Arrow.
    - CLI highlights:
      - `--subset {solutions_py,solutions_w_editorials_py}` or `--subset-name` to override
      - `--match-strategy {basic,robust}` with `--abs-tol`, `--rel-tol`
      - `--include-generated-tests` fallback when no public/private tests
      - `--dataset-split`, `--streaming` for HF loading
      - `--failed-output-dir` to save failed samples
  - `decontaminate_converted_cots.py`, `parallel_filter_solution_py.py`: complementary preprocessing tools.

- `deepcoder_gen_data/`
  - `convert_trajectories_to_cots.py`: convert successful solution of Deepcoder trajectories as judged by the reward model to CoT format
  - `run_rllm_from_parquet.py`: run Deepcoder trajectory generation from parquet files and prepare training data

- `compare_datasets/`
  - `compare_datasets.py`, `compare_single_problem.py`: quick diffs/inspection across datasets (Mixture-of-Thought and Deepcoder trajectories) or specific problems

- `visualize/`
  - `visualize_cots_datasets_web.py`, `visualize_single_problem_pair.py`: lightweight web/CLI visualizers

### Quickstart: Filtering High-Quality CoT Samples

Run with robust matching and generated-test fallback:

```bash
python datasets/preprocess_data/filter_and_save_cots.py \
  --subset solutions_py \
  --match-strategy robust \
  --endpoint http://localhost:2000 \
  --output-path ./codeforces_cots_high_quality.arrow \
  --num-workers 16
```

Alternate subset and deterministic matching example:

```bash
python datasets/preprocess_data/filter_and_save_cots.py \
  --subset solutions_w_editorials_py \
  --match-strategy basic \
  --include-generated-tests=false \
  --dataset-split 'train[:200]'
```

### Dependencies

- `datasets`, `tqdm`, `requests` (for Piston API)
- Python 3.10+

### Tips

- Ensure your execution server is available at `--endpoint` (e.g., Piston) before running filters.
- Use `--failed-output-dir` to capture failing examples for analysis and UI parity during visualization.
- For very large runs, prefer `--streaming` and increase `--num-workers` according to your cores and service limits.


