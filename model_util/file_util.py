#!/usr/bin/env python3
"""
model_util.py

Shared utilities for training and inference scripts.
Focus: readability, reuse, and consistent behavior across scripts.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from typing import Any, List, Tuple, Optional

from transformers import AutoConfig, AutoTokenizer


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sanitize_for_filename(text: str) -> str:
    t = (text or "").strip().lower()
    t = t.replace("/", "-")
    t = t.replace(" ", "_")
    t = re.sub(r"[^a-z0-9._-]", "", t)
    t = re.sub(r"[-_]{2,}", "-", t)
    return t or "model"


def resolve_outfile(model_name: str, dataset_name: str, subset: str, split: str) -> str:
    model_tag = sanitize_for_filename(model_name)
    ds_tag = sanitize_for_filename(dataset_name)
    subset_tag = sanitize_for_filename(subset)
    split_tag = sanitize_for_filename(split)
    return f"{model_tag}__{ds_tag}__{subset_tag}__{split_tag}__vllm.jsonl"


def latest_checkpoint(run_dir: str) -> str | None:
    try:
        if not os.path.isdir(run_dir):
            return None
        cks = [d for d in os.listdir(run_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(run_dir, d))]
        if not cks:
            return None
        cks_sorted = sorted(cks, key=lambda n: int(re.sub(r"[^0-9]", "", n)) if re.search(r"\d+", n) else -1)
        return os.path.join(run_dir, cks_sorted[-1])
    except Exception:
        return None


def is_valid_trainer_checkpoint_dir(path: str) -> bool:
    try:
        if not os.path.isdir(path):
            return False
        state_file = os.path.join(path, "trainer_state.json")
        return os.path.isfile(state_file)
    except Exception:
        return False


def latest_valid_trainer_checkpoint(parent_dir: str) -> Optional[str]:
    try:
        cand = [p for p in os.listdir(parent_dir) if p.startswith("checkpoint-")]
        cand_full = [os.path.join(parent_dir, p) for p in cand if os.path.isdir(os.path.join(parent_dir, p))]
        cand_sorted = sorted(
            cand_full,
            key=lambda p: int(p.rsplit("-", 1)[-1]) if p.rsplit("-", 1)[-1].isdigit() else -1,
            reverse=True,
        )
        for ck in cand_sorted:
            if is_valid_trainer_checkpoint_dir(ck):
                return ck
    except Exception:
        pass
    return None


def normalize_single_file(src_dir: str, dst_dir: str) -> bool:
    os.makedirs(dst_dir, exist_ok=True)
    copied = False
    for name in ["pytorch_model.bin", "model.safetensors"]:
        p = os.path.join(src_dir, name)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(dst_dir, name))
            copied = True
    # Nested bug case: dir named pytorch_model.bin/
    nested = os.path.join(src_dir, "pytorch_model.bin")
    if os.path.isdir(nested):
        inner_bin = os.path.join(nested, "pytorch_model.bin")
        inner_safe = os.path.join(nested, "model.safetensors")
        if os.path.isfile(inner_bin):
            shutil.copy2(inner_bin, os.path.join(dst_dir, "pytorch_model.bin"))
            copied = True
        elif os.path.isfile(inner_safe):
            shutil.copy2(inner_safe, os.path.join(dst_dir, "model.safetensors"))
            copied = True
    if copied:
        for fname in [
            "config.json","generation_config.json","tokenizer.json","tokenizer_config.json",
            "special_tokens_map.json","vocab.json","merges.txt","added_tokens.json","chat_template.jinja",
        ]:
            s = os.path.join(src_dir, fname)
            if os.path.isfile(s):
                try:
                    shutil.copy2(s, os.path.join(dst_dir, fname))
                except Exception:
                    pass
    return copied


def prepare_model_dir(checkpoint_path: str, model_name: str, ds_convert: Optional[Any]) -> str:
    # 1) No checkpoint_path: return model_name
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return model_name

    path = checkpoint_path
    # If a run dir with checkpoint-*/ exists, pick latest
    if os.path.isdir(path) and any(n.startswith("checkpoint-") for n in os.listdir(path)):
        ck = latest_checkpoint(path)
        if ck:
            path = ck

    # If this looks like a DeepSpeed checkpoint (has global_step*/ inside)
    if os.path.isdir(path) and any(n.startswith("global_step") for n in os.listdir(path)):
        if ds_convert is None:
            print("Warning: deepspeed not available; cannot auto-convert ZeRO checkpoint. Falling back to model_name.")
            return model_name
        tmpdir = tempfile.mkdtemp(prefix="vllm_ds2hf_")
        out_file = os.path.join(tmpdir, "pytorch_model.bin")
        try:
            print(f"Auto-converting DeepSpeed checkpoint: {path} -> {out_file}")
            ds_convert(path, out_file)
            # If ds_convert creates a directory for sharded weights, normalize the structure.
            if os.path.isdir(out_file):
                print(f"Normalizing sharded checkpoint structure created at: {out_file}")
                for item in os.listdir(out_file):
                    shutil.move(os.path.join(out_file, item), os.path.join(tmpdir, item))
                os.rmdir(out_file)
            # Save tokenizer and config alongside for completeness
            try:
                tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                tok.save_pretrained(tmpdir)
            except Exception:
                pass
            try:
                conf = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                conf.save_pretrained(tmpdir)
            except Exception:
                pass
            return tmpdir
        except Exception as e:
            print(f"Auto-convert failed: {e}")
            shutil.rmtree(tmpdir)
            return model_name

    # If directory already has a single-file weight or nested bug, normalize
    tmpdir = tempfile.mkdtemp(prefix="vllm_norm_")
    if normalize_single_file(path, tmpdir):
        return tmpdir

    # If it's an HF sharded directory with index at root, vLLM can usually load directly
    index_root = os.path.join(path, "pytorch_model.bin.index.json")
    if os.path.isfile(index_root):
        return path

    # If shards are nested under pytorch_model.bin/, normalize them upward (copy index/shards)
    nested_index = os.path.join(path, "pytorch_model.bin", "pytorch_model.bin.index.json")
    if os.path.isfile(nested_index):
        try:
            # Copy all files from nested into tmpdir
            for n in os.listdir(os.path.dirname(nested_index)):
                s = os.path.join(os.path.dirname(nested_index), n)
                d = os.path.join(tmpdir, n)
                if os.path.isfile(s):
                    shutil.copy2(s, d)
            # Copy tokenizer/config from root
            for fname in [
                "config.json","generation_config.json","tokenizer.json","tokenizer_config.json",
                "special_tokens_map.json","vocab.json","merges.txt","added_tokens.json","chat_template.jinja",
            ]:
                s = os.path.join(path, fname)
                if os.path.isfile(s):
                    shutil.copy2(s, os.path.join(tmpdir, fname))
            return tmpdir
        except Exception:
            pass

    # Otherwise fall back to model_name
    return model_name


