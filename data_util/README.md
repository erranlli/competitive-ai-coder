## data_util: Shared Utilities for Benchmarks and Notebooks

Reusable helpers for problem lookup, rich HTML rendering, and evaluation visualization. These utilities are imported by the notebooks in `benchmark/` and anywhere else you need quick, readable views of data.

### Modules

- `codeforces.py`
  - `get_record_by_problem_id(ds, problem_id)`
  - `find_records_by_contest(ds, contest_id)`
  - `list_available_problem_ids(ds, limit=50)`
  - `search_problems_by_title(ds, title_keyword, limit=10)`
  - `quick_problem_lookup(ds, problem_id)`
  - `explore_contest(ds, contest_id)`
  - `pretty_print_codeforces_problem(record)` and `pretty_print_codeforces_problem_dark(record)`

- `piston_eval.py`
  - `load_jsonl_to_dict(filepath)` and `load_json_records(filepath)`
  - `find_record_by_problem_id(records, problem_id)` and `list_all_problem_ids(records)`
  - Pretty-printers for model outputs and evaluation comparisons:
    - `pretty_print_model_output(record)`
    - `pretty_print_single_record(record, passedTests=True)`
    - `pretty_print_piston_results_all(records, passedTests=True, limit=10)`
    - `pretty_print_piston_results(json_lines, passedTests=True)`
    - `display_test_results(file_path, problem_id=None)`

- `programming_pretty.py`
  - `pretty_print_programming_record(record, record_type="competitive_programming")`
  - `pretty_print_programming_record_veri(record, record_type="competitive_programming")`

All functions are re-exported via `data_util/__init__.py` for convenient imports.

### Quickstart

In notebooks or scripts, ensure the repo root is on `sys.path`:

```python
import sys
sys.path.insert(0, "/root/competitive-coding-ai")

from data_util import (
  get_record_by_problem_id, pretty_print_codeforces_problem_dark,
  load_jsonl_to_dict, display_test_results,
)
```

Use with any Hugging Face dataset split that has Codeforces fields:

```python
from datasets import load_dataset
ds = load_dataset("open-r1/codeforces", split="test")
rec = get_record_by_problem_id(ds, "2063/C")
pretty_print_codeforces_problem_dark(rec)
```

Visualize evaluation JSONL outputs:

```python
sol = load_jsonl_to_dict("/path/to/solutions.jsonl")
display_test_results("/path/to/piston_eval_results.jsonl", problem_id="2063/C")
```

### Notes

- Renderers output rich HTML; they are best viewed in Jupyter notebooks.
- If you edit these utilities, re-run imports in your notebook to pick up changes.


