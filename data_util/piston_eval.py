import html
import json
from typing import List, Dict, Any, Optional
from IPython.display import display, HTML


def load_jsonl_to_dict(filepath: str) -> Dict[str, Any]:
    import os
    if not os.path.exists(filepath):
        print(f"❌ Error: File not found at '{filepath}'")
        return {}
    data_dict: Dict[str, Any] = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    record = json.loads(line)
                    record_id = record.get("id")
                    if record_id:
                        data_dict[record_id] = record
                    else:
                        print(f"⚠️ Warning: Record on line {i+1} has no 'id' field. Skipping.")
                except json.JSONDecodeError:
                    print(f"⚠️ Warning: Could not decode JSON on line {i+1}. Skipping.")
        return data_dict
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return {}


def pretty_print_model_output(record: Dict[str, Any]) -> None:
    if not isinstance(record, dict):
        print("❌ Error: The provided record is not a valid dictionary.")
        return
    try:
        from pygments import highlight
        from pygments.lexers import get_lexer_by_name
        from pygments.formatters import HtmlFormatter
        formatter = HtmlFormatter(style='monokai')
        pygments_css = formatter.get_style_defs('.highlight')
        lexer = get_lexer_by_name('python')
        code = record.get("code", "# No code found")
        highlighted_code = highlight(code, lexer, HtmlFormatter(style='monokai'))
    except Exception:
        pygments_css = ""
        code = html.escape(record.get("code", "# No code found"))
        highlighted_code = f'<div class="highlight" style="background: #272822; color: #f8f8f2"><pre><code>{code}</code></pre></div>'

    problem_id = record.get("id", "N/A")
    model = record.get("model", "Unknown Model")
    time_limit = record.get("time_limit", "?")
    memory_limit = record.get("memory_limit", "?")
    timestamp = record.get("timestamp")
    from datetime import datetime
    readable_time = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S UTC") if timestamp else "N/A"

    styles = f"""
    <style>
        .code-gen-container {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", sans-serif;
            border: 1px solid #d0d7de;
            border-radius: 8px;
            margin-bottom: 24px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            overflow: hidden;
        }}
        .header {{
            display: flex; justify-content: space-between; align-items: center; padding: 12px 20px;
            background-color: #f6f8fa; border-bottom: 1px solid #d0d7de;
        }}
        .problem-id {{ font-size: 20px; font-weight: 600; color: #1f2328; }}
        .model-name {{
            font-family: monospace, sans-serif; font-size: 12px; padding: 4px 8px;
            background-color: #e6f7ff; color: #0366d6; border: 1px solid #91d5ff; border-radius: 6px;
        }}
        .metadata {{
            padding: 10px 20px;
            border-bottom: 1px solid #d0d7de;
            font-size: 13px;
            color: #57606a;
            display: flex;
            gap: 24px;
        }}
        .metadata strong {{ color: #24292e; }}
        .code-block {{ background-color: #272822; }}
        .code-block pre {{ margin: 0; padding: 16px; white-space: pre-wrap; word-wrap: break-word; background-color: #272822; color: #f8f8f2; }}
        .highlight {{ background-color: #272822 !important; }}
        .footer {{ padding: 8px 20px; font-size: 12px; color: #8c959d; text-align: right; background-color: #f6f8fa; border-top: 1px solid #d0d7de; }}
        {pygments_css}
    </style>
    """

    full_html = styles + f"""
    <div class="code-gen-container">
        <div class="header">
            <span class="problem-id">{html.escape(problem_id)}</span>
            <span class="model-name">{html.escape(model)}</span>
        </div>
        <div class="metadata">
            <span><strong>Time Limit:</strong> {time_limit}s</span>
            <span><strong>Memory Limit:</strong> {memory_limit} MB</span>
        </div>
        <div class="code-block">
            {highlighted_code}
        </div>
        <div class="footer">
            <span>Generated on: {readable_time}</span>
        </div>
    </div>
    """
    display(HTML(full_html))


def load_json_records(file_path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        records.append(record)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse JSON on line {line_num}: {e}")
        print(f"Successfully loaded {len(records)} records from {file_path}")
        return records
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return []
    except Exception as e:
        print(f"Error loading file: {e}")
        return []


def find_record_by_problem_id(records: List[Dict[str, Any]], problem_id: str) -> Optional[Dict[str, Any]]:
    for record in records:
        if record.get("problem_id") == problem_id:
            return record
    return None


def list_all_problem_ids(records: List[Dict[str, Any]]):
    return [record.get("problem_id", "Unknown") for record in records]


def pretty_print_single_record(record: Optional[Dict[str, Any]], passedTests: bool = True) -> None:
    if not record:
        print("No record found!")
        return
    json_line = json.dumps(record)
    pretty_print_piston_results(json_line, passedTests)


def pretty_print_piston_results_all(records: List[Dict[str, Any]], passedTests: bool = True, limit: int = 10) -> None:
    cnt = 0
    for record in records:
        status = record.get("status", "unknown")
        inOutput = not passedTests or (passedTests and (status == "True" or status != "failed"))
        if inOutput:
            cnt += 1
            pretty_print_single_record(record, passedTests)
        if cnt >= limit:
            break


def pretty_print_piston_results(json_data: str, passedTests: bool = True) -> None:
    styles = """
    <style>
        .result-container { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", sans-serif; border: 1px solid #d0d7de; border-radius: 6px; margin-bottom: 24px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); background-color: #ffffff; }
        .result-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #d0d7de; padding-bottom: 8px; margin-bottom: 16px; background-color: #ffffff; }
        .problem-id { font-size: 20px; font-weight: 600; color: #1f2328; }
        .status-badge { padding: 5px 12px; border-radius: 2em; font-weight: 600; font-size: 14px; }
        .status-passed { color: #1f883d; background-color: #dafbe1; }
        .status-failed { color: #cf222e; background-color: #ffebe9; }
        .detail-container { border: 1px solid #d8dee4; border-radius: 6px; margin-top: 15px; overflow: hidden; background-color: #ffffff; }
        .detail-header { background-color: #fafbfc; padding: 8px 12px; border-bottom: 1px solid #d8dee4; font-weight: 600; color: #1f2328; }
        .detail-header-passed { border-left: 4px solid #2da44e; }
        .detail-header-failed { border-left: 4px solid #d1272f; }
        .detail-body { padding: 15px; background-color: #ffffff; }
        .comparison-table { width: 100%; border-collapse: collapse; table-layout: fixed; background-color: #ffffff; }
        .comparison-table th, .comparison-table td { border: 1px solid #d0d7de; padding: 8px; text-align: left; vertical-align: top; white-space: pre-wrap; word-wrap: break-word; font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace; font-size: 13px; color: #1f2328; background-color: #ffffff; }
        .comparison-table th { background-color: #fafbfc; color: #1f2328; }
        .comparison-table tr.mismatch-row { background-color: #fff8dc; }
        .comparison-table tr.mismatch-row td { color: #8b0000; font-weight: 600; border-color: #ff6b6b; background-color: #fff8dc; }
        .comparison-table .line-num { color: #8c959d; text-align: right; width: 40px; background-color: #fafbfc; }
        .comparison-table tr.mismatch-row .line-num { background-color: #ffebe9; }
        .error-log { background-color: #fafbfc; color: #24292e; padding: 15px; border: 1px solid #d0d7de; border-radius: 6px; white-space: pre-wrap; word-wrap: break-word; font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace; font-size: 13px; }
    </style>
    """
    full_html = styles
    for line in json_data.strip().split('\n'):
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            continue
        problem_id = result.get("problem_id", "N/A")
        status = result.get("status", "unknown")
        status_class = "status-passed" if status == "passed" else "status-failed"
        if passedTests and status_class == "status-failed":
            continue
        full_html += f'<div class="result-container">'
        full_html += f'<div class="result-header">'
        full_html += f'<span class="problem-id">{problem_id}</span>'
        full_html += f'<span class="status-badge {status_class}">{status.upper()}</span>'
        full_html += f'</div>'
        for i, detail in enumerate(result.get("details", [])):
            case_passed = detail.get("passed", False)
            header_class = "detail-header-passed" if case_passed else "detail-header-failed"
            case_status = "PASSED" if case_passed else "FAILED"
            full_html += f'<div class="detail-container">'
            full_html += f'<div class="detail-header {header_class}">Test Case #{i+1} ({detail.get("case_type", "N/A")}) &mdash; {case_status}</div>'
            full_html += f'<div class="detail-body">'
            error = detail.get("error")
            if error:
                full_html += f'<h5>Error Output:</h5><pre class="error-log">{html.escape(error)}</pre>'
            else:
                output = detail.get("output", "").split('\n')
                expected = detail.get("expected", "").replace('\r\n', '\n').split('\n')
                if output and output[-1] == '' and expected and expected[-1] == '':
                    output.pop()
                    expected.pop()
                max_lines = max(len(output), len(expected))
                full_html += '<table class="comparison-table">'
                full_html += '<thead><tr><th class="line-num">#</th><th>Your Output</th><th>Expected Output</th></tr></thead>'
                full_html += '<tbody>'
                for j in range(max_lines):
                    output_line = output[j] if j < len(output) else ""
                    expected_line = expected[j] if j < len(expected) else ""
                    row_class = "mismatch-row" if output_line != expected_line else ""
                    full_html += f'<tr class="{row_class}">' \
                                 f'<td class="line-num">{j+1}</td>' \
                                 f'<td class="{"your-output" if output_line != expected_line else ""}">{html.escape(output_line)}</td>' \
                                 f'<td>{html.escape(expected_line)}</td>' \
                                 f'</tr>'
                full_html += '</tbody></table>'
            full_html += f'</div></div>'
        full_html += '</div>'
    display(HTML(full_html))


def display_test_results(file_path: str, problem_id: Optional[str] = None) -> None:
    records = load_json_records(file_path)
    if not records:
        return
    if problem_id:
        record = find_record_by_problem_id(records, problem_id)
        if record:
            print(f"🎨 Displaying record for problem ID: '{problem_id}'")
            pretty_print_single_record(record)
        else:
            print(f"❌ Problem ID '{problem_id}' not found!")
            print("Available problem IDs:")
            for pid in list_all_problem_ids(records):
                print(f"  - {pid}")
    else:
        print(f"🎨 Displaying all {len(records)} records")
        json_lines = [json.dumps(record) for record in records]
        pretty_print_piston_results('\n'.join(json_lines))


