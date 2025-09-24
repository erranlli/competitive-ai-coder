
from __future__ import annotations

import re
from typing import Any, List, Tuple, Optional


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(line.rstrip() for line in lines).strip()


def extract_code_from_text(generated: str, language_hint: str = "python") -> str:
    if generated is None:
        return ""
    fence_pattern = re.compile(r"```(?:" + re.escape(language_hint) + r"|\w+)?\n([\s\S]*?)```", re.IGNORECASE)
    match = fence_pattern.search(generated)
    if match:
        return (match.group(1) or "").strip()
    any_fence = re.compile(r"```\n([\s\S]*?)```")
    match = any_fence.search(generated)
    if match:
        return (match.group(1) or "").strip()
    return generated.strip()
