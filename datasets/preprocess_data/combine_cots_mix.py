#!/usr/bin/env python3
"""
Combine two CoT-style datasets into a mixed set:

- Keep ALL samples from the first dataset
- From the second dataset:
  - Keep only samples with token length >= threshold (default: 16000)
  - Drop any sample that has an 8-gram overlap with the first dataset

Saves the result via datasets.save_to_disk to the output directory (default: ./cot_deepcoder_mix).
"""

import argparse
import os
import sys
from typing import Any, Dict, Iterable, List, Set, Tuple

from datasets import load_from_disk, Dataset, concatenate_datasets
from transformers import AutoTokenizer


# Ensure repository root is on sys.path for local package imports
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_CUR_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from model_util.text_util import extract_code_from_text, normalize_text  # noqa: E402


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def extract_assistant_text(row: Dict[str, Any]) -> str:
    msgs = row.get("messages")
    if isinstance(msgs, list) and len(msgs) >= 2:
        try:
            if (msgs[0].get("role") == "user") and (msgs[1].get("role") == "assistant"):
                return str(msgs[1].get("content") or "")
        except Exception:
            pass
    # fallbacks
    for key in ("generation", "output", "answer", "final_code", "code"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def text_for_overlap(row: Dict[str, Any]) -> str:
    asst = extract_assistant_text(row)
    code = extract_code_from_text(asst, language_hint="python")
    return code if code.strip() else asst


def whitespace_tokens(text: str) -> List[str]:
    # normalize lightly to improve matching
    t = normalize_text(text).lower()
    return [tok for tok in t.split() if tok]


def ngrams(tokens: List[str], n: int = 8) -> Iterable[Tuple[str, ...]]:
    if len(tokens) < n:
        return []
    for i in range(0, len(tokens) - n + 1):
        yield tuple(tokens[i : i + n])


def collect_8gram_set(ds: Dataset) -> Set[Tuple[str, ...]]:
    grams: Set[Tuple[str, ...]] = set()
    for row in ds:
        txt = text_for_overlap(row)
        toks = whitespace_tokens(txt)
        grams.update(ngrams(toks, 8))
    return grams


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Combine two CoT datasets into a mixed set with dedup and length filtering")
    p.add_argument("--first-ds", required=True, help="Path to the first dataset (kept entirely)")
    p.add_argument("--second-ds", required=True, help="Path to the second dataset (filtered)")
    p.add_argument("--output-dir", default="./cot_deepcoder_mix", help="Output directory for the combined dataset")
    p.add_argument("--tokenizer-model", default="Qwen/Qwen2.5-7B-Instruct", help="Model name for tokenization")
    p.add_argument("--length-threshold", type=int, default=16000, help="Minimum token length to keep from the second dataset")
    p.add_argument("--num-proc", type=int, default=max(1, (os.cpu_count() or 4) - 2), help="Processes for map/filter")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    first_path = os.path.abspath(args.first_ds)
    second_path = os.path.abspath(args.second_ds)
    out_path = os.path.abspath(args.output_dir)

    if not os.path.isdir(first_path):
        raise SystemExit(f"First dataset directory not found: {first_path}")
    if not os.path.isdir(second_path):
        raise SystemExit(f"Second dataset directory not found: {second_path}")

    print(f"Loading first dataset: {first_path}")
    ds1 = load_from_disk(first_path)
    print(f"First dataset rows: {len(ds1)}")

    print(f"Loading second dataset: {second_path}")
    ds2 = load_from_disk(second_path)
    print(f"Second dataset rows: {len(ds2)}")

    # Build n-gram set from the first dataset
    print("Collecting 8-gram set from the first dataset for dedup...")
    grams1 = collect_8gram_set(ds1)
    print(f"Collected {len(grams1)} unique 8-grams from dataset 1")

    # Tokenizer for length thresholding
    print(f"Loading tokenizer: {args.tokenizer_model}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer_model, trust_remote_code=True)

    def compute_token_len(row: Dict[str, Any]) -> Dict[str, Any]:
        try:
            msgs = row.get("messages")
            if isinstance(msgs, list) and len(msgs) >= 1:
                full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            else:
                full = extract_assistant_text(row)
            ids = tok(full, add_special_tokens=False)["input_ids"]
            row["_token_len"] = len(ids)
        except Exception:
            row["_token_len"] = 0
        return row

    print("Computing token lengths for the second dataset...")
    ds2_len = ds2.map(compute_token_len, desc="Tokenizing", num_proc=max(1, args.num_proc))
    ds2_len = ds2_len.filter(lambda r: int(r.get("_token_len", 0)) >= int(args.length_threshold), desc="Length filter")
    print(f"After length filter (>= {args.length_threshold}), rows: {len(ds2_len)}")

    def no_overlap_with_first(row: Dict[str, Any]) -> bool:
        try:
            txt = text_for_overlap(row)
            toks = whitespace_tokens(txt)
            for gram in ngrams(toks, 8):
                if gram in grams1:
                    return False
            return True
        except Exception:
            return True

    print("Removing rows from second dataset with 8-gram overlap against first dataset...")
    ds2_filtered = ds2_len.filter(no_overlap_with_first, desc="8-gram dedup vs first")
    print(f"Second dataset after dedup: {len(ds2_filtered)}")

    # Combine
    combined = concatenate_datasets([ds1, ds2_filtered])
    print(f"Combined rows: {len(combined)}")

    # Save
    if os.path.exists(out_path):
        print(f"Output path exists, overwriting: {out_path}")
        # best-effort cleanup
        import shutil as _sh
        _sh.rmtree(out_path)
    ensure_dir(out_path)
    combined.save_to_disk(out_path)
    print(f"Saved combined dataset to: {out_path}")


if __name__ == "__main__":
    main()


