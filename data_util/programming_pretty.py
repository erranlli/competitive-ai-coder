from typing import Dict
from IPython.display import display, HTML
import html


def pretty_print_programming_record(record: Dict, record_type: str = "competitive_programming", *, truncate_chars: int = 500, show_cases: int = 3) -> None:
    if not isinstance(record, dict):
        print("❌ Invalid record")
        return
    sections = []
    title = record.get("title") or record.get("name") or record.get("id") or record_type
    sections.append(f"<h2>{html.escape(str(title))}</h2>")
    meta_pairs = []
    for k in ["id", "rating", "time_limit", "memory_limit", "tags", "div", "problem_id"]:
        if k in record and record[k] is not None:
            meta_pairs.append(f"<b>{html.escape(k)}:</b> {html.escape(str(record[k]))}")
    if meta_pairs:
        sections.append("<p>" + "<br>".join(meta_pairs) + "</p>")
    # Long text fields
    for field in [
        "description", "problem-description",
        "input", "input-specification",
        "output", "output-specification",
        "note",
    ]:
        if field in record and record[field]:
            text = str(record[field])
            shown = text if len(text) <= truncate_chars else (text[:truncate_chars] + "...")
            sections.append(f"<h3>{html.escape(field)}</h3><div>{html.escape(shown)}</div>")
    # Examples / tests
    examples = None
    if "examples" in record:
        examples = record["examples"]
    elif "test_cases" in record:
        examples = record["test_cases"]
    if examples:
        items = []
        for i, ex in enumerate(examples[:show_cases]):
            if isinstance(ex, dict):
                inp = html.escape(str(ex.get("input", "")))
                out = html.escape(str(ex.get("output", "")))
            elif isinstance(ex, (list, tuple)) and len(ex) >= 2:
                inp = html.escape(str(ex[0]))
                out = html.escape(str(ex[1]))
            else:
                inp = html.escape(str(ex))
                out = ""
            items.append(f"<div style='border:1px solid #ddd;padding:8px;margin:6px;'><b>Case {i+1}</b><br><b>Input:</b><pre>{inp}</pre><b>Output:</b><pre>{out}</pre></div>")
        sections.append("<h3>Examples</h3>" + "".join(items))
    display(HTML("".join(sections)))


def pretty_print_programming_record_veri(record: Dict, record_type: str = "competitive_programming") -> None:
    # Thin wrapper for compatibility with notebooks
    pretty_print_programming_record(record, record_type)


