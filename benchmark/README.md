## Benchmark: Codeforces Exploration and Visualization

This directory contains notebooks and artifacts to explore Codeforces-style datasets, visualize problems and solutions, and prepare small JSON datasets for quick sanity checks.

### Contents

- `r1_codeforces.ipynb`: End-to-end exploration for the `open-r1/codeforces` dataset.
  - Loads train/test splits (and verifiable subset)
  - Pretty-prints problems using shared utilities in `data_util`
  - Compares model solution JSONL files and evaluation results
  - Uses repo-relative paths via a small helper cell to locate `model_solutions/` and `bench_results/`

- `rllm_codeforces.ipynb`: Lightweight dataset builder for quick demos.
  - Reads Codeforces problems from public datasets
  - Emits small train/test JSON files under `train/code/` and `test/code/` for sandboxing

- `train/code/` and `test/code/`: Generated JSON artifacts from the notebooks (safe to regenerate).

### Prerequisites

- Python 3.10+
- The project root must be on `sys.path` so that shared utilities resolve:
  - Both notebooks insert the repository root (`/root/competitive-coding-ai`) at runtime
  - Shared helpers live in `data_util/` and are imported by the notebooks

Recommended pip environment (already used by the repo):

```bash
pip install datasets pandas ipython
```

### Shared Utilities

Common functions for pretty-printing, record lookup, and eval visualization live under `data_util/`:

- `data_util/codeforces.py`: problem lookup and rich HTML pretty-printers
- `data_util/piston_eval.py`: load JSONL results, pretty-print per-record comparisons, list IDs
- `data_util/programming_pretty.py`: generic programming record pretty-printers

In the notebooks, these are imported via:

```python
import sys
sys.path.insert(0, "/root/competitive-coding-ai")
from data_util import (
  get_record_by_problem_id,
  pretty_print_codeforces_problem_dark,
  load_jsonl_to_dict,
  display_test_results,
)
```

### Repo-relative Paths

`r1_codeforces.ipynb` uses a small helper to locate the repository root by searching for the `model_solutions/` folder. This avoids hard-coding absolute paths.

```python
from pathlib import Path

def find_repo_root(marker: str = "model_solutions") -> Path:
    p = Path.cwd()
    for parent in [p, *p.parents]:
        if (parent / marker).exists():
            return parent
    return p

REPO_ROOT = find_repo_root()
sol_path = REPO_ROOT / "model_solutions" / "<solutions.jsonl>"
```

Place your solution JSONL files under `model_solutions/` and evaluation results under `bench_results/` to keep paths consistent.

### Typical Workflows

- Inspect a specific Codeforces problem:
  1. Open `r1_codeforces.ipynb`
  2. Run the first two setup cells
  3. Use `get_record_by_problem_id(test_ds, "2063/C")`
  4. Render with `pretty_print_codeforces_problem_dark(record)`

- Compare solution JSONL and evaluation records:
  - Load solutions with `load_jsonl_to_dict(<path>)`
  - Load eval results with `load_json_records(<path>)`
  - Select problems: `problems = ["2030/D", "2063/C", ...]`
  - Pretty-print using utilities from `data_util/piston_eval.py`

- Generate a small demo dataset:
  - Open `rllm_codeforces.ipynb`
  - Run all cells; outputs appear under `train/code/` and `test/code/`

### Troubleshooting

- ModuleNotFoundError: `data_util`
  - Ensure the first cell that does `sys.path.insert(0, "/root/competitive-coding-ai")` ran.

- Missing solution IDs (KeyError)
  - The notebook now guards against missing keys and prints which problem IDs are absent from the loaded JSONL.

- Paths not found
  - Verify `model_solutions/` and `bench_results/` exist at the repository root. Adjust the helper function if you use a different layout.

### Notes

- Notebooks are intended for exploration and visualization; they do not modify the training datasets directly.
- If you change the shared utilities in `data_util/`, re-run the first import cell in the notebooks to reload modules.


