import argparse
import webbrowser
import os
from threading import Timer
from flask import Flask, render_template_string, abort
from datasets import load_from_disk
import mistune
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import html
from pygments.util import ClassNotFound

# --- Configuration ---
# DATASET_A_PATH = "/mnt/data2/filtered_datasets_flexible_match/successful_solutions"
DATASET_A_PATH = "/mnt/data2/codeforces_cots_high_quality.arrow"
DATASET_B_PATH = "/mnt/data2/new_deepcoder_cots_arrow_appexp"

# --- HTML Template ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Problem Comparison</title>
    <style>
        body { font-family: sans-serif; line-height: 1.6; margin: 0; padding: 0; background-color: #f4f4f4; color: #333; }
        .container { display: flex; max-width: 95%; margin: 20px auto; }
        .column { flex: 1; padding: 20px; margin: 10px; background-color: #fff; border: 1px solid #ddd; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1, h2, h3 { color: #333; }
        h1 { text-align: center; margin-bottom: 20px; }
        h2 { border-bottom: 2px solid #eee; padding-bottom: 10px; }
        h3 { color: #555; font-size: 1.1em; }
        pre { white-space: pre-wrap; word-wrap: break-word; background-color: #2b2b2b; color: #f8f8f2; padding: 15px; border-radius: 5px; }
        code { font-family: 'Courier New', Courier, monospace; }
        .prompt { background-color: #eef; padding: 15px; border-radius: 5px; border: 1px solid #dde; }
    </style>
</head>
<body>
    <h1>Problem Comparison</h1>
    <div class="container">
        <div class="column">
            <h2>{{ ds_a.name }}</h2>
            <h3>ID: {{ ds_a.id }}</h3>
            <div class="content">
                <h3>User Prompt</h3>
                <div class="prompt">{{ ds_a.prompt | safe }}</div>
                <h3>Assistant Response</h3>
                <div>{{ ds_a.response | safe }}</div>
            </div>
        </div>
        <div class="column">
            <h2>{{ ds_b.name }}</h2>
            <h3>ID: {{ ds_b.id }}</h3>
            <div class="content">
                <h3>User Prompt</h3>
                <div class="prompt">{{ ds_b.prompt | safe }}</div>
                <h3>Assistant Response</h3>
                <div>{{ ds_b.response | safe }}</div>
            </div>
        </div>
    </div>
</body>
</html>
"""

# --- Markdown and Syntax Highlighting ---
class HighlightRenderer(mistune.HTMLRenderer):
    def block_code(self, code, info=None):
        if info:
            try:
                lexer = get_lexer_by_name(info, stripall=True)
                formatter = html.HtmlFormatter()
                return highlight(code, lexer, formatter)
            except ClassNotFound:
                # If the language alias is not found, fallback to a plain code block.
                pass
        return '<pre><code>' + mistune.escape(code) + '</code></pre>'

markdown = mistune.create_markdown(renderer=HighlightRenderer(escape=False))

# --- Flask App ---
app = Flask(__name__)
problem_data = {}

def get_problem(ds, problem_id):
    """Fetches a single problem record from a dataset."""
    try:
        # Use filter which is generally efficient for HuggingFace datasets
        record = ds.filter(lambda example: example['id'] == problem_id, num_proc=4)[0]
        messages = record.get('messages', [{}, {}])
        
        prompt_raw = messages[0].get('content', 'N/A')
        prompt_html = markdown(prompt_raw) # Apply markdown rendering to the prompt
        
        response_raw = messages[1].get('content', 'N/A')
        response_html = markdown(response_raw)
        
        return {'id': problem_id, 'prompt': prompt_html, 'response': response_html}
    except IndexError:
        return None
    except Exception as e:
        print(f"Error fetching problem {problem_id}: {e}")
        return None

@app.route('/')
def compare_problems_view():
    if not problem_data.get('ds_a') or not problem_data.get('ds_b'):
        abort(404, "Problem data not found. Check server logs for errors.")
    return render_template_string(HTML_TEMPLATE, ds_a=problem_data['ds_a'], ds_b=problem_data['ds_b'])

def main():
    parser = argparse.ArgumentParser(description="Visualize a pair of problems from two datasets in a web browser.")
    parser.add_argument("--id_a", type=str, required=True, help="ID of the problem from Dataset A.")
    parser.add_argument("--id_b", type=str, required=True, help="ID of the problem from Dataset B.")
    parser.add_argument("--port", type=int, default=5001, help="Port to run the web server on.")
    args = parser.parse_args()

    print("Loading datasets...")
    try:
        ds_a = load_from_disk(DATASET_A_PATH, keep_in_memory=False)
        ds_b = load_from_disk(DATASET_B_PATH, keep_in_memory=False)
        print("Datasets loaded successfully.")
    except Exception as e:
        print(f"Fatal: Could not load datasets. Error: {e}")
        return

    print(f"Fetching problem '{args.id_a}' from Dataset A...")
    problem_a_data = get_problem(ds_a, args.id_a)
    if not problem_a_data:
        print(f"Fatal: Could not find or load problem '{args.id_a}'.")
        return
    problem_a_data['name'] = os.path.basename(DATASET_A_PATH)
    problem_data['ds_a'] = problem_a_data

    print(f"Fetching problem '{args.id_b}' from Dataset B...")
    problem_b_data = get_problem(ds_b, args.id_b)
    if not problem_b_data:
        print(f"Fatal: Could not find or load problem '{args.id_b}'.")
        return
    problem_b_data['name'] = os.path.basename(DATASET_B_PATH)
    problem_data['ds_b'] = problem_b_data
    
    url = f"http://127.0.0.1:{args.port}"
    print(f"Starting server... Open this URL in your browser: {url}")
    Timer(1, lambda: webbrowser.open_new_tab(url)).start()
    
    app.run(port=args.port, debug=False)

if __name__ == "__main__":
    main()
