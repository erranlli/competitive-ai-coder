#!/usr/bin/env python3
"""
decontaminate_converted_cots.py

Filters out problems from a converted COTS-style dataset that also appear in
the open-r1 Codeforces test split. We reuse the idea of robust text
normalization and ID-based matching, and optionally apply n-gram overlap
filtering inspired by open-r1 decontamination
(see: https://github.com/huggingface/open-r1/blob/main/scripts/decontaminate.py).

Input: a HuggingFace dataset saved via save_to_disk (from the converter).
Output: a new dataset directory (save_to_disk) with overlapping problems removed.

Matching rules (union):
- ID match: any candidate identifier (id, contest_id/index, problem_id) matches test set.
- Text match: normalized question text hash matches test set.
- Optional n-gram rule: exclude if sample's question shares any n-gram (default 8-gram)
  with any test question.

Usage:
  python scripts/decontaminate_converted_cots.py \
    --input-path /path/to/converted_arrow_dir \
    --output-path /path/to/decontaminated_arrow_dir \
    [--test-dataset open-r1/codeforces] [--test-split test] \
    [--use-ngram-overlap --ngram-size 8]
"""

import argparse
import hashlib
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from datasets import load_dataset, load_from_disk, Dataset


def _normalize_text_basic(text: str) -> str:
    if text is None:
        return ""
    # strip trailing spaces in lines and trim
    lines = str(text).splitlines()
    return "\n".join(line.rstrip() for line in lines).strip()


def _text_hash(text: str) -> str:
    t = _normalize_text_basic(text)
    return hashlib.sha1(t.encode("utf-8")).hexdigest()


def _tokenize(text: str) -> List[str]:
    # Lowercase, keep alphanumerics and underscore as word tokens
    return re.findall(r"\w+", _normalize_text_basic(text).lower())


def _ngrams(tokens: Iterable[str], n: int) -> Set[Tuple[str, ...]]:
    toks: List[str] = list(tokens)
    if n <= 0 or len(toks) < n:
        return set()
    return {tuple(toks[i : i + n]) for i in range(0, len(toks) - n + 1)}


def _candidate_ids(rec: Dict[str, Any]) -> Set[str]:
    cands: Set[str] = set()
    # direct id
    rid = rec.get("id")
    if isinstance(rid, str) and rid:
        cands.add(rid)

    # composed from contest/index
    contest_id = rec.get("contest_id")
    index = rec.get("index")
    if contest_id is not None and index is not None:
        cands.add(f"{contest_id}/{index}")
        cands.add(f"{contest_id}-{index}")

    # problem_id
    problem_id = rec.get("problem_id")
    if problem_id is not None:
        cands.add(str(problem_id))

    return {c for c in cands if isinstance(c, str) and c}


def _question_text(rec: Dict[str, Any]) -> Optional[str]:
    # try common keys in priority order
    for key in ("question", "statement", "problem", "content", "prompt", "title"):
        val = rec.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return None


def _extract_problem_core(text: str) -> str:
    """Heuristically strip generic system prompt scaffolding and keep core spec.

    - Prefer content starting at "solve the following coding problem".
    - Stop before sections like "### format" / "### answer" if present.
    - If markers not found, fallback to original text after removing obvious headers.
    """
    if not isinstance(text, str):
        return ""
    s = text
    s_lower = s.lower()

    # Find start marker
    start_idx = 0
    start_markers = [
        "solve the following coding problem",
        "problem:",
    ]
    for m in start_markers:
        pos = s_lower.find(m)
        if pos != -1:
            start_idx = pos
            break

    # Find end marker
    end_idx = len(s)
    end_markers = [
        "### format",
        "### answer",
    ]
    for m in end_markers:
        pos = s_lower.find(m)
        if pos != -1:
            end_idx = min(end_idx, pos)

    core = s[start_idx:end_idx]

    # Remove fenced code blocks if any accidentally included
    core = re.sub(r"```[\s\S]*?```", " ", core)
    return _normalize_text_basic(core)


def build_test_sets(test_dataset_name: str, test_split: str) -> Tuple[Set[str], Set[str]]:
    test_ds = load_dataset(test_dataset_name, split=test_split)
    test_ids: Set[str] = set()
    test_qhash: Set[str] = set()

    for rec in test_ds:
        for cid in _candidate_ids(rec):
            test_ids.add(cid)
        q = _question_text(rec)
        if q:
            test_qhash.add(_text_hash(q))

    return test_ids, test_qhash


def main():
    parser = argparse.ArgumentParser(description="Decontaminate converted COTS dataset against open-r1 Codeforces test set.")
    parser.add_argument("--input-path", required=True, help="Path to dataset saved via save_to_disk")
    parser.add_argument("--output-path", required=True, help="Where to save the filtered dataset (save_to_disk)")
    parser.add_argument("--test-dataset", default="open-r1/codeforces", help="Test dataset HF hub name")
    parser.add_argument("--test-split", default="test", help="Test split to use")
    parser.add_argument("--use-ngram-overlap", action="store_true", help="Enable n-gram overlap decontamination rule vs test set")
    parser.add_argument("--ngram-size", type=int, default=8, help="n-gram size for overlap (default: 8)")
    parser.add_argument("--dedupe-mode", choices=["ngram", "none"], default="none", help="In-dataset deduplication mode: ngram or none (default: none)")
    parser.add_argument("--dedupe-ngram-size", type=int, default=8, help="n-gram size for in-dataset deduplication (default: 8)")
    parser.add_argument("--ngram-stop-threshold", type=float, default=0.01, help="Ignore n-grams that appear in more than this fraction of samples (default: 0.01)")
    args = parser.parse_args()

    print(f"Loading input dataset from {args.input_path} ...")
    ds = load_from_disk(args.input_path)
    print(f"Input size: {len(ds)}")

    print(f"Loading test set {args.test_dataset}:{args.test_split} ...")
    test_ids, test_qhash = build_test_sets(args.test_dataset, args.test_split)
    print(f"Test IDs: {len(test_ids)} | Test question hashes: {len(test_qhash)}")

    # Optional: precompute test question n-grams
    test_ngram_set: Set[Tuple[str, ...]] = set()
    if args.use_ngram_overlap:
        print(f"Building test {args.ngram_size}-gram set for overlap filtering ...")
        test_ds_for_ngrams = load_dataset(args.test_dataset, split=args.test_split)
        for rec in test_ds_for_ngrams:
            q = _question_text(rec)
            if not q:
                continue
            test_ngram_set |= _ngrams(_tokenize(q), args.ngram_size)
        print(f"Test {args.ngram_size}-gram set size: {len(test_ngram_set)}")

    def keep_fn(rec: Dict[str, Any]) -> bool:
        # ID-based exclusion
        for cid in _candidate_ids(rec):
            if cid in test_ids:
                return False
        # Text-based exclusion
        q = _question_text(rec)
        if q and _text_hash(q) in test_qhash:
            return False
        # n-gram-based exclusion
        if args.use_ngram_overlap and q:
            cand = _ngrams(_tokenize(q), args.ngram_size)
            if cand and test_ngram_set.intersection(cand):
                return False
        return True

    print("Filtering...")
    filtered = ds.filter(keep_fn, num_proc=os.cpu_count() or 1)
    print(f"Kept {len(filtered)} of {len(ds)} examples.")

    # Optional in-dataset deduplication
    deduped = filtered
    if args.dedupe_mode == "ngram":
        print(f"Deduplicating by {args.dedupe_ngram_size}-gram overlap within the dataset (order-preserving) with common n-grams ignored...")

        # Precompute common n-grams to ignore
        n = args.dedupe_ngram_size
        counts: Dict[Tuple[str, ...], int] = {}
        texts: List[str] = []
        for i in range(len(filtered)):
            q_raw = _question_text(filtered[i])
            core = _extract_problem_core(q_raw) if q_raw else ""
            texts.append(core)
            grams = _ngrams(_tokenize(core), n)
            for g in grams:
                counts[g] = counts.get(g, 0) + 1
        cutoff = max(1, int(args.ngram_stop_threshold * len(filtered)))
        common = {g for g, c in counts.items() if c > cutoff}
        print(f"Ignoring {len(common)} common n-grams (threshold>{cutoff} occurrences)")

        seen_ngrams: Set[Tuple[str, ...]] = set()
        keep_indices: list[int] = []
        for i, core in enumerate(texts):
            grams = _ngrams(_tokenize(core), n)
            grams = grams - common
            if grams and seen_ngrams.intersection(grams):
                continue
            keep_indices.append(i)
            seen_ngrams |= grams

        deduped = filtered.select(keep_indices)
        print(f"Deduped size: {len(deduped)} (removed {len(filtered) - len(deduped)})")

    print(f"Saving to {args.output_path} ...")
    deduped.save_to_disk(args.output_path)
    print("Done.")


if __name__ == "__main__":
    main()


