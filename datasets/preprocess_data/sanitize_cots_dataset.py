#!/usr/bin/env python3
"""
sanitize_cots_dataset.py

Parallel sanitization for COTS-style datasets saved with datasets.save_to_disk.

Defaults sanitize fields:
- generation: string with assistant output (may include <think> and code)
- messages: list[dict] with {'role','content'}; only 'assistant' contents are sanitized

Usage:
  python datasets/preprocess_data/sanitize_cots_dataset.py \
    --input-path /mnt/data2/new_deepcoder_cots_arrow_appexp \
    --output-path /mnt/data2/new_deepcoder_cots_arrow_appexp_sanitized \
    --num-proc 16
"""

import argparse
import os
import re
from typing import Any, Dict, List

from datasets import load_from_disk, Dataset, DatasetDict


def sanitize_response(assistant_response: str) -> str:
    # Replace repetitive "Wait, no. Wait, " patterns with a single clean "Wait, "
    return re.sub(r"(\bWait, no\.\s*Wait,\s*)", "Wait, ", assistant_response, flags=re.IGNORECASE)


def sanitize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    updated: Dict[str, Any] = {}

    gen = row.get("generation")
    if isinstance(gen, str) and gen:
        updated["generation"] = sanitize_response(gen)

    msgs = row.get("messages")
    if isinstance(msgs, list):
        new_msgs: List[Any] = []
        for m in msgs:
            if isinstance(m, dict):
                role = m.get("role")
                content = m.get("content")
                if role == "assistant" and isinstance(content, str) and content:
                    nm = dict(m)
                    nm["content"] = sanitize_response(content)
                    new_msgs.append(nm)
                else:
                    new_msgs.append(m)
            else:
                new_msgs.append(m)
        updated["messages"] = new_msgs

    # Return only mutated columns to keep schema stable
    return updated


def process_dataset(ds: Dataset, num_proc: int) -> Dataset:
    return ds.map(sanitize_row, desc="Sanitizing", num_proc=max(1, num_proc))


def main() -> None:
    ap = argparse.ArgumentParser(description="Parallel sanitize a saved HF dataset (save_to_disk format)")
    ap.add_argument("--input-path", required=True, help="Path to dataset saved via save_to_disk")
    ap.add_argument("--output-path", required=True, help="Destination path for sanitized dataset")
    ap.add_argument("--num-proc", type=int, default=os.cpu_count() or 4, help="Number of processes for map()")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output path if it exists")
    args = ap.parse_args()

    in_path = os.path.abspath(args.input_path)
    out_path = os.path.abspath(args.output_path)

    if os.path.exists(out_path):
        if not args.overwrite:
            raise SystemExit(f"Output path exists: {out_path}. Use --overwrite to replace.")
        # Best-effort cleanup
        import shutil
        shutil.rmtree(out_path)

    data = load_from_disk(in_path)

    if isinstance(data, DatasetDict):
        new_dict = DatasetDict()
        for split, split_ds in data.items():
            new_dict[split] = process_dataset(split_ds, args.num_proc)
        new_dict.save_to_disk(out_path)
    elif isinstance(data, Dataset):
        new_ds = process_dataset(data, args.num_proc)
        new_ds.save_to_disk(out_path)
    else:
        raise SystemExit(f"Unsupported dataset type: {type(data)}")

    print(f"Sanitized dataset saved to: {out_path}")


if __name__ == "__main__":
    main()


