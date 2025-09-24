#!/usr/bin/env python3
"""
model_util.py

Shared utilities for training and inference scripts.
Focus: readability, reuse, and consistent behavior across scripts.
"""

from __future__ import annotations

from typing import Any, Tuple

from transformers import AutoConfig, AutoTokenizer


def get_model_config_fields(model_name: str) -> Tuple[int, int, int]:
    try:
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        num_heads = int(getattr(cfg, "num_attention_heads", 0) or 0)
        num_kv_heads = int(getattr(cfg, "num_key_value_heads", 0) or 0)
        max_len = int(getattr(cfg, "max_position_embeddings", 0) or 0)
        return num_heads, num_kv_heads, max_len
    except Exception:
        return 0, 0, 0


def detect_model_type(model_name: str) -> str:
    model_lower = (model_name or "").lower()
    if "deepseek-r1" in model_lower:
        return "deepseek-r1"
    if "qwen2.5" in model_lower:
        return "qwen2.5"
    if "qwen3" in model_lower:
        return "qwen3"
    return "generic"


def load_tokenizer(model_name: str) -> Any:
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if getattr(tok, "pad_token", None) is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok

