#!/usr/bin/env python3
"""
visualize_cots_datasets_web.py

Gradio UI to visualize and compare two COTS-style Arrow datasets.
Shows:
- Dataset features and column lists
- First N samples (default 20), selectable, and renders EVERY column with names
- Column diff between the two datasets

Usage:
  python scripts/visualize_cots_datasets_web.py \
    --path-a /mnt/data2/codeforces_cots_high_quality.arrow \
    --path-b /mnt/data2/deepcoder_cots_arrow \
    --num-samples 20 \
    --port 12346
"""

import argparse
import json
import re
from typing import Any, Dict, List

import gradio as gr
from datasets import load_from_disk, Dataset


def format_value(value: Any, max_chars: int = 4000) -> str:
    try:
        if isinstance(value, (dict, list)):
            s = json.dumps(value, ensure_ascii=False, indent=2)
        else:
            s = str(value)
    except Exception:
        s = str(value)
    if max_chars and len(s) > max_chars:
        return s[:max_chars] + f"\n... [truncated {len(s) - max_chars} chars]"
    return s


def summarize_types(ds: Dataset, num_samples: int = 200) -> Dict[str, List[str]]:
    limit = min(len(ds), num_samples)
    type_map: Dict[str, set] = {col: set() for col in ds.column_names}
    for i in range(limit):
        row = ds[i]
        for col in ds.column_names:
            v = row.get(col)
            tname = type(v).__name__
            type_map[col].add(tname)
    return {k: sorted(list(v)) for k, v in type_map.items()}


def _max_backtick_run(text: str) -> int:
    mx = 0
    cur = 0
    for ch in text:
        if ch == '`':
            cur += 1
            if cur > mx:
                mx = cur
        else:
            cur = 0
    return mx


def wrap_in_safe_fence(text: str) -> str:
    # Use a fence length longer than any backtick run in the content
    longest = _max_backtick_run(text)
    fence = "`" * (longest + 1 if longest >= 3 else 3)
    return f"{fence}\n{text}\n{fence}"


def render_dataset_header(name: str, ds: Dataset) -> str:
    header = [f"## Dataset {name}", f"- Path loaded", f"- Num rows: {len(ds)}", f"- Columns: {ds.column_names}"]
    header.append("\n### Features")
    header.append(f"```")
    header.append(str(ds.features))
    header.append("```")
    header.append("\n### Column Type Summary (first 200 rows)")
    types = summarize_types(ds)
    for col in ds.column_names:
        header.append(f"- {col}: {types.get(col, [])}")
    return "\n".join(header)


def render_sample(ds: Dataset, index: int) -> str:
    if len(ds) == 0:
        return "*Empty dataset*"
    index = max(0, min(index, len(ds) - 1))
    row = ds[index]
    parts = [f"### Sample {index+1}/{len(ds)}"]
    for col in ds.column_names:
        parts.append(f"\n**[{col}]**:\n")
        # Avoid truncation for rich fields
        max_chars = None if col in {"generation", "messages"} else 4000
        content = format_value(row.get(col), max_chars=max_chars)
        parts.append(wrap_in_safe_fence(content))
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser(description="Gradio visualizer for two COTS-style datasets")
    #ap.add_argument("--path-a", default="/mnt/data2/new_codeforces_cots_high_quality_arrow")
    ap.add_argument("--path-a", default="/mnt/data2/new_codeforces_cots_failed_arrow")
    ap.add_argument("--path-b", default="/mnt/data2/new_deepcoder_cots_arrow_appexp")
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--port", type=int, default=12347)
    ap.add_argument("--ids-a", type=str, default=None, help="Comma-separated list of ids to include for dataset A (e.g., '1509/B,1469/B')")
    ap.add_argument("--ids-a-csv-file", type=str, default=None, help="Path to a file containing ids for dataset A (comma/newline-separated; accepts tokens like 'id=1509/B')")
    args = ap.parse_args()

    ds_a = load_from_disk(args.path_a)
    ds_b = load_from_disk(args.path_b)

    # Optional filtering of dataset A by IDs
    id_set: set[str] = set()

    def _normalize_id_token(token: str) -> str | None:
        t = (token or "").strip()
        if not t:
            return None
        if t.startswith("id="):
            t = t[3:].strip()
        # Accept forms like 1509/B, 324/E2, 690/D3
        if re.match(r"^\d+/[A-Za-z0-9]+$", t):
            return t
        return None

    if args.ids_a:
        try:
            for tok in re.split(r"[\s,]+", args.ids_a):
                nt = _normalize_id_token(tok)
                if nt:
                    id_set.add(nt)
        except Exception:
            pass
    if args.ids_a_csv_file:
        try:
            with open(args.ids_a_csv_file, 'r', encoding='utf-8') as f:
                content = f.read()
            # Extract explicit id=... tokens
            for tok in re.findall(r"id=(\d+/[A-Za-z0-9]+)", content):
                id_set.add(tok)
            # Also parse bare tokens separated by commas/whitespace
            for tok in re.split(r"[\s,]+", content):
                nt = _normalize_id_token(tok)
                if nt:
                    id_set.add(nt)
        except Exception:
            pass
    if id_set:
        pre_rows = len(ds_a)
        try:
            if 'id' in ds_a.column_names:
                # More efficient and robust: pass only 'id' as input column
                ds_a = ds_a.filter(lambda id: id in id_set, input_columns=['id'])
            else:
                # Fallback: try dict-style
                ds_a = ds_a.filter(lambda ex: (ex.get('id') if isinstance(ex, dict) else None) in id_set)
        except Exception:
            try:
                ds_a = ds_a.filter(lambda ex: (ex.get('id') if isinstance(ex, dict) else ex['id']) in id_set)
            except Exception:
                pass
        post_rows = len(ds_a)
        print(f"Filtered dataset A by {len(id_set)} ids -> kept {post_rows}/{pre_rows} rows")

    # Limit to first N samples for easy browsing (but if ids were provided, show only those ids without truncation)
    n_a = min(len(ds_a), args.num_samples)
    n_b = min(len(ds_b), args.num_samples)
    ds_a_small = ds_a if len(id_set) > 0 else (ds_a.select(range(n_a)) if n_a > 0 else ds_a)
    ds_b_small = ds_b.select(range(n_b)) if n_b > 0 else ds_b

    cols_a = set(ds_a.column_names)
    cols_b = set(ds_b.column_names)
    only_a = sorted(list(cols_a - cols_b))
    only_b = sorted(list(cols_b - cols_a))

    # Gradio app
    with gr.Blocks(title="COTS Dataset Visualizer") as demo:
        gr.Markdown("# 📊 COTS Dataset Visualizer")

        # Headers / feature summaries
        header_a = gr.Markdown(render_dataset_header("A", ds_a))
        header_b = gr.Markdown(render_dataset_header("B", ds_b))

        gr.Markdown("## Column Differences")
        gr.Markdown(f"**Only in A:** {only_a}\n\n**Only in B:** {only_b}")

        with gr.Row():
            with gr.Column():
                gr.Markdown("## A: Samples")
                idx_a = gr.Slider(minimum=1, maximum=max(1, n_a), step=1, value=1, label="A Sample Index (1-based)")
                out_a = gr.Markdown()
            with gr.Column():
                gr.Markdown("## B: Samples")
                idx_b = gr.Slider(minimum=1, maximum=max(1, n_b), step=1, value=1, label="B Sample Index (1-based)")
                out_b = gr.Markdown()

        def on_idx_a(i: int) -> str:
            return render_sample(ds_a_small, int(i) - 1)

        def on_idx_b(i: int) -> str:
            return render_sample(ds_b_small, int(i) - 1)

        idx_a.change(fn=on_idx_a, inputs=idx_a, outputs=out_a)
        idx_b.change(fn=on_idx_b, inputs=idx_b, outputs=out_b)

        # Initialize
        demo.load(fn=lambda: render_sample(ds_a_small, 0), inputs=None, outputs=out_a)
        demo.load(fn=lambda: render_sample(ds_b_small, 0), inputs=None, outputs=out_b)

    demo.launch(share=True, server_port=args.port)


if __name__ == "__main__":
    main()


