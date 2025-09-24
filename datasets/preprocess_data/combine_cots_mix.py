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
import re
import keyword
import builtins as _builtins
from collections import defaultdict
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
    # Prefer the LAST assistant turn to avoid shared system/user prompts
    if isinstance(msgs, list) and msgs:
        try:
            for m in reversed(msgs):
                if isinstance(m, dict) and (m.get("role") == "assistant"):
                    content = m.get("content")
                    if isinstance(content, str) and content.strip():
                        return content
                    # Some datasets store content as a list of parts
                    if isinstance(content, list) and content:
                        try:
                            parts: List[str] = []
                            for part in content:
                                if isinstance(part, dict):
                                    text = part.get("text") or part.get("content") or ""
                                    if isinstance(text, str) and text:
                                        parts.append(text)
                                elif isinstance(part, str):
                                    parts.append(part)
                            joined = "\n".join([p for p in parts if p.strip()])
                            if joined.strip():
                                return joined
                        except Exception:
                            pass
        except Exception:
            pass
    # Fallbacks for non-chat schemas
    for key in ("generation", "output", "answer", "final_code", "code"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def text_for_overlap(row: Dict[str, Any]) -> str:
    asst = extract_assistant_text(row)
    code = extract_code_from_text(asst, language_hint="python")
    return code if code.strip() else asst


_PY_STOP = set(kw.lower() for kw in keyword.kwlist)
_PY_STOP.update(name for name in dir(_builtins) if name.isidentifier())
_PY_STOP.update({
    # common boilerplate identifiers in CF/CP code
    "main", "solve", "test", "tests", "case", "cases", "input", "output",
    "stdin", "stdout", "data", "read", "write", "sys", "os", "math",
})


def _is_boilerplate_line(line: str) -> bool:
    l = line.strip()
    if not l:
        return True
    if re.match(r"^(?:from\s+\S+\s+import\s+\S+|import\s+\S+)", l):
        return True
    if re.match(r"^if\s+__name__\s*==\s*['\"]__main__['\"]\s*:\s*$", l):
        return True
    if re.match(r"^def\s+main\s*\(.*\)\s*:\s*$", l):
        return True
    if re.match(r"^sys\.setrecursionlimit\(.*\)\s*$", l):
        return True
    if re.match(r"^threading\.stack_size\(.*\)\s*$", l):
        return True
    return False


def tokens_for_overlap(text: str) -> List[str]:
    # Prefer code when available; remove comments and boilerplate
    base = text or ""
    # Strip Python comments
    base = re.sub(r"(?m)#.*$", "", base)
    # Remove blatant boilerplate lines
    lines = [ln for ln in base.splitlines() if not _is_boilerplate_line(ln)]
    base = "\n".join(lines)
    # Extract identifiers only, ignore short/common tokens; keep digits-only long tokens to capture constants
    idents = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\b\d{3,}\b", base)
    toks = []
    for tok in idents:
        low = tok.lower()
        if len(low) < 3:
            continue
        if low in _PY_STOP:
            continue
        toks.append(low)
    return toks


def ngrams(tokens: List[str], n: int = 8) -> Iterable[Tuple[str, ...]]:
    if len(tokens) < n:
        return []
    for i in range(0, len(tokens) - n + 1):
        yield tuple(tokens[i : i + n])


def collect_8gram_index(ds: Dataset) -> Dict[Tuple[str, ...], Set[int]]:
    index: Dict[Tuple[str, ...], Set[int]] = defaultdict(set)
    for idx, row in enumerate(ds):
        txt = text_for_overlap(row)
        toks = tokens_for_overlap(txt)
        for gram in ngrams(toks, 8):
            index[gram].add(idx)
    return index


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Combine two CoT datasets into a mixed set with dedup and length filtering")
    p.add_argument("--first-ds", required=True, help="Path to the first dataset (kept entirely)")
    p.add_argument("--second-ds", required=True, help="Path to the second dataset (filtered)")
    p.add_argument("--output-dir", default="./cot_deepcoder_mix", help="Output directory for the combined dataset")
    p.add_argument("--tokenizer-model", default="Qwen/Qwen2.5-7B-Instruct", help="Model name for tokenization")
    p.add_argument("--max-len", type=int, default=16384, help="Maximum token length to keep from the second dataset")
    p.add_argument("--num-proc", type=int, default=max(1, (os.cpu_count() or 4) - 2), help="Processes for map/filter")
    p.add_argument("--length-mode", choices=["assistant", "assistant_code", "full"], default="assistant", help="Which text to measure for token length filtering")
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

    # Build n-gram inverted index from the first dataset, and gate by document-level single-use
    print("Indexing 8-grams from the first dataset for dedup (assistant-only, code-aware)...")
    grams_index = collect_8gram_index(ds1)
    print(f"Indexed {len(grams_index)} unique 8-grams from dataset 1")
    # Each document in ds1 can eliminate at most one row from ds2
    used_docs: Set[int] = set()

    # Tokenizer for length thresholding
    print(f"Loading tokenizer: {args.tokenizer_model}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer_model, trust_remote_code=True)

    def compute_token_len(row: Dict[str, Any]) -> Dict[str, Any]:
        try:
            mode = args.length_mode
            if mode == "full":
                msgs = row.get("messages")
                if isinstance(msgs, list) and len(msgs) >= 1:
                    full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                else:
                    full = extract_assistant_text(row)
            elif mode == "assistant_code":
                full = extract_code_from_text(extract_assistant_text(row), language_hint="python")
                if not full.strip():
                    full = extract_assistant_text(row)
            else:  # "assistant"
                full = extract_assistant_text(row)
            ids = tok(full, add_special_tokens=False)["input_ids"]
            row["_token_len"] = len(ids)
        except Exception:
            row["_token_len"] = 0
        return row

    print(f"Computing token lengths for the second dataset (mode: {args.length_mode})...")
    ds2_len = ds2.map(compute_token_len, desc="Tokenizing", num_proc=max(1, args.num_proc))
    ds2_len = ds2_len.filter(lambda r: int(r.get("_token_len", 0)) <= int(args.max_len), desc="Length filter")
    print(f"After length filter (<= {args.max_len}), rows: {len(ds2_len)}")

    dropped_count = {"n": 0}

    def no_overlap_with_first(row: Dict[str, Any]) -> bool:
        try:
            txt = text_for_overlap(row)
            toks = tokens_for_overlap(txt)
            for gram in ngrams(toks, 8):
                doc_ids = grams_index.get(gram)
                if not doc_ids:
                    continue
                # Find an unused doc that contains this gram
                for d in doc_ids:
                    if d not in used_docs:
                        used_docs.add(d)
                        dropped_count["n"] += 1
                        return False
            return True
        except Exception:
            return True

    print("Removing rows from second dataset with 8-gram overlap against first dataset...")
    ds2_filtered = ds2_len.filter(no_overlap_with_first, desc="8-gram dedup vs first")
    print(f"Second dataset after dedup: {len(ds2_filtered)} (dropped {dropped_count['n']} due to overlaps; max drops allowed = {len(ds1)}; length_mode={args.length_mode})")

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


