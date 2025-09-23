#!/usr/bin/env python3
"""
Gradio web viewer for large JSONL files.

- Lists all .jsonl files under a target directory
- Streams and indexes records (line offsets) without loading entire file
- Browse by index; filter by substring in problem_id/title if present
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr


def safe_json_dumps(obj: Any, max_chars: int = 8000) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        try:
            s = str(obj)
        except Exception:
            s = "<unprintable>"
    if max_chars and len(s) > max_chars:
        return s[:max_chars] + f"\n... [truncated {len(s)-max_chars} chars]"
    return s


def build_index(file_path: str, max_lines: Optional[int] = None) -> Tuple[List[int], int]:
    offsets: List[int] = []
    total = 0
    with open(file_path, "rb") as f:
        pos = f.tell()
        line_no = 0
        for line in f:
            offsets.append(pos)
            line_no += 1
            if max_lines and line_no >= max_lines:
                break
            pos = f.tell()
        total = line_no
    return offsets, total


def read_record_at(file_path: str, byte_offset: int) -> Dict[str, Any]:
    with open(file_path, "rb") as f:
        f.seek(byte_offset)
        raw = f.readline()
    try:
        return json.loads(raw.decode("utf-8", errors="replace"))
    except Exception:
        return {"_raw": raw.decode("utf-8", errors="replace").rstrip("\n")}


def format_record(rec: Dict[str, Any]) -> str:
    # Highlight common fields if present
    parts: List[str] = []
    for key in ["id", "problem_id", "title", "pass", "status", "language"]:
        if key in rec:
            parts.append(f"**{key}:** {rec[key]}")
    # Show code/solution fields with fences if present
    code_keys = [
        "solution", "generated_code", "answer", "final_code", "pred",
        "candidate", "program", "code"
    ]
    for ck in code_keys:
        if ck in rec and isinstance(rec[ck], str):
            parts.append(f"### {ck}\n\n```python\n{rec[ck]}\n```")

    # Render raw_response with sections if present
    raw_resp = rec.get("raw_response")
    if isinstance(raw_resp, str) and raw_resp.strip():
        # Extract <think> blocks
        try:
            think_blocks = re.findall(r"<think>([\s\S]*?)</think>", raw_resp, flags=re.IGNORECASE)
        except Exception:
            think_blocks = []

        # Extract the first fenced code block
        code_snippet = None
        try:
            fence_lang = re.compile(r"```(?:python|\w+)?\n([\s\S]*?)```", re.IGNORECASE)
            m = fence_lang.search(raw_resp)
            if m:
                code_snippet = (m.group(1) or "").strip()
        except Exception:
            code_snippet = None

        parts.append("### raw_response (rendered)")
        if think_blocks:
            tb = (think_blocks[0] or "").strip()
            parts.append("#### Thinking\n\n" + (f"```\n{tb}\n```" if tb else "<empty>"))
        if code_snippet:
            parts.append("#### Code (from raw_response)\n\n```python\n" + code_snippet + "\n```")
        # Include full raw_response so fences render correctly
        parts.append("#### raw_response (full)\n\n" + raw_resp)

    # Fallback: full JSON
    parts.append("### Full Record\n\n" + safe_json_dumps(rec))
    return "\n\n".join(parts)


def find_first_index_matching(
    file_path: str, offsets: List[int], needle: str
) -> int:
    needle_lower = needle.lower().strip()
    if not needle_lower:
        return 0
    for i, off in enumerate(offsets):
        rec = read_record_at(file_path, off)
        text_fields = [
            str(rec.get("problem_id", "")),
            str(rec.get("id", "")),
            str(rec.get("title", "")),
        ]
        joined = " \n ".join(text_fields).lower()
        if needle_lower in joined:
            return i
    return 0


def main():
    ap = argparse.ArgumentParser(description="Visualize JSONL files in a directory via Gradio")
    ap.add_argument("--dir", required=True, help="Directory containing .jsonl files")
    ap.add_argument("--port", type=int, default=12355)
    ap.add_argument("--index-cap", type=int, default=None, help="Optional cap on number of lines to index per file")
    args = ap.parse_args()

    root_dir = os.path.abspath(args.dir)
    if not os.path.isdir(root_dir):
        raise SystemExit(f"Directory not found: {root_dir}")

    jsonl_files = [
        os.path.join(root_dir, n)
        for n in sorted(os.listdir(root_dir))
        if n.endswith(".jsonl")
    ]

    if not jsonl_files:
        raise SystemExit(f"No .jsonl files in {root_dir}")

    # State
    state_selected: Dict[str, Any] = {
        "file": jsonl_files[0],
        "offsets": [],
        "total": 0,
    }

    def load_file(selected_path: str) -> Tuple[str, int, int, str]:
        offsets, total = build_index(selected_path, args.index_cap)
        state_selected["file"] = selected_path
        state_selected["offsets"] = offsets
        state_selected["total"] = total
        pretty_name = os.path.basename(selected_path)
        return pretty_name, 1 if total > 0 else 0, max(1, total), (format_record(read_record_at(selected_path, offsets[0])) if total > 0 else "No records")

    def on_index_change(i: int) -> str:
        total = state_selected["total"]
        if total <= 0:
            return "No records"
        i0 = max(1, min(int(i), total)) - 1
        off = state_selected["offsets"][i0]
        rec = read_record_at(state_selected["file"], off)
        return format_record(rec)

    def on_filter(text: str) -> Tuple[int, str]:
        total = state_selected["total"]
        if total <= 0:
            return 0, "No records"
        idx0 = find_first_index_matching(state_selected["file"], state_selected["offsets"], text)
        off = state_selected["offsets"][idx0]
        rec = read_record_at(state_selected["file"], off)
        return idx0 + 1, format_record(rec)

    with gr.Blocks(title="JSONL Viewer") as demo:
        gr.Markdown("# 📄 JSONL Viewer")
        dd = gr.Dropdown(choices=jsonl_files, value=jsonl_files[0], label="File")
        name = gr.Markdown()
        with gr.Row():
            idx = gr.Slider(1, 1, value=1, step=1, label="Record Index (1-based)")
            total_md = gr.Number(value=0, label="Total", interactive=False, precision=0)
        filt = gr.Textbox(label="Find first record with substring in id/problem_id/title")
        out = gr.Markdown()

        dd.change(fn=load_file, inputs=dd, outputs=[name, idx, total_md, out])
        idx.change(fn=on_index_change, inputs=idx, outputs=out)
        filt.submit(fn=on_filter, inputs=filt, outputs=[idx, out])

        # initialize
        init_name, init_idx, init_total, init_out = load_file(jsonl_files[0])
        name.value = f"**File:** {init_name}"
        idx.maximum = max(1, init_total)
        idx.value = init_idx
        total_md.value = init_total
        out.value = init_out

    demo.launch(share=True, server_port=args.port)


if __name__ == "__main__":
    main()


