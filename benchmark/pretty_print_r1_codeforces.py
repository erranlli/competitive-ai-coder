import html
import json
from datetime import datetime
from IPython.display import display, HTML
import os

try:
    from pygments import highlight
    from pygments.lexers import get_lexer_by_name
    from pygments.formatters import HtmlFormatter
    PYGMENTS_AVAILABLE = True
except Exception:
    PYGMENTS_AVAILABLE = False
    
def get_record_by_problem_id(ds, problem_id):
    """
    Get a record from the dataset by problem ID (format: 'contest_id/index').
    Uses the 'id' field directly for lookup.
    """
    try:
        for record in ds:
            if record.get('id') == problem_id:
                return dict(record)
        return None
    except Exception as e:
        print(f"Error searching for problem_id '{problem_id}': {e}")
        return None


def find_records_by_contest(ds, contest_id):
    """
    Get all records from a specific contest.
    
    Args:
        ds: The HuggingFace dataset
        contest_id: Contest ID (integer or string)
        
    Returns:
        List of dictionaries containing matching records
    """
    try:
        contest_id = int(contest_id)  # Ensure it's an integer
        records = []
        
        for record in ds:
            if record['contest_id'] == contest_id:
                records.append(dict(record))
        
        return records
        
    except ValueError:
        print(f"Invalid contest_id: '{contest_id}'. Must be a number.")
        return []
    except Exception as e:
        print(f"Error searching for contest_id '{contest_id}': {e}")
        return []

def list_available_problem_ids(ds, limit=50):
    """
    List available problem IDs in the dataset.
    
    Args:
        ds: The HuggingFace dataset
        limit: Maximum number of problem IDs to display (default: 50)
        
    Returns:
        List of problem IDs in format 'contest_id/index'
    """
    problem_ids = []
    
    for i, record in enumerate(ds):
        if i >= limit:
            break
        # Handle both string and integer contest_id
        contest_id = record['contest_id']
        problem_id = f"{contest_id}/{record['index']}"
        problem_ids.append(problem_id)
    
    return problem_ids

def search_problems_by_title(ds, title_keyword, limit=10):
    """
    Search for problems by title keyword.
    
    Args:
        ds: The HuggingFace dataset
        title_keyword: Keyword to search for in titles
        limit: Maximum number of results to return
        
    Returns:
        List of tuples: (problem_id, title)
    """
    results = []
    title_keyword = title_keyword.lower()
    
    for record in ds:
        if len(results) >= limit:
            break
            
        title = record.get('title', '').lower()
        if title_keyword in title:
            problem_id = f"{record['contest_id']}/{record['index']}"
            results.append((problem_id, record.get('title', 'No title')))
    
    return results

def display_problem_info(record):
    """
    Display key information about a problem record in a nice format.
    
    Args:
        record: Problem record dictionary
    """
    if not record:
        print("No record found!")
        return
    
    problem_id = f"{record.get('contest_id', 'Unknown')}/{record.get('index', 'Unknown')}"
    
    print("=" * 60)
    print(f"🎯 Problem ID: {problem_id}")
    print(f"📝 Title: {record.get('title', 'No title')}")
    print(f"🏆 Contest: {record.get('contest_name', 'Unknown')} ({record.get('contest_type', 'Unknown')})")
    print(f"📅 Year: {record.get('contest_start_year', 'Unknown')}")
    if record.get('rating'):
        print(f"⭐ Rating: {record.get('rating')}")
    if record.get('tags'):
        print(f"🏷️  Tags: {', '.join(record.get('tags', []))}")
    print(f"⏱️  Time Limit: {record.get('time_limit', 'Unknown')}")
    print(f"💾 Memory Limit: {record.get('memory_limit', 'Unknown')}")
    print("=" * 60)
    
    if record.get('description'):
        print("📖 Description:")
        print(record['description'][:500] + ("..." if len(record['description']) > 500 else ""))
        print()
    
    if record.get('examples'):
        print("💡 Examples:")
        examples = record['examples']
        if isinstance(examples, list) and len(examples) > 0:
            for i, example in enumerate(examples[:2]):  # Show first 2 examples
                print(f"  Example {i+1}:")
                if isinstance(example, dict):
                    if 'input' in example:
                        print(f"    Input: {example['input']}")
                    if 'output' in example:
                        print(f"    Output: {example['output']}")
                else:
                    print(f"    {example}")
                print()

# Example usage functions:

def quick_problem_lookup(ds, problem_id):
    """
    Quick lookup and display of a problem.
    
    Usage:
        quick_problem_lookup(ds, '2063/B')
    """
    record = get_record_by_problem_id(ds, problem_id)
    display_problem_info(record)
    return record

def explore_contest(ds, contest_id):
    """
    Explore all problems in a contest.
    
    Usage:
        explore_contest(ds, 2063)
    """
    records = find_records_by_contest(ds, contest_id)
    print(f"Found {len(records)} problems in contest {contest_id}:")
    print()
    
    for record in records:
        problem_id = f"{record['contest_id']}/{record['index']}"
        title = record.get('title', 'No title')
        rating = record.get('rating', 'Unrated')
        print(f"  {problem_id}: {title} (Rating: {rating})")
    
    return records

# Usage examples:
"""
# Load your dataset first (replace with your actual dataset loading code)
# ds = load_dataset('your_dataset_name')

# Get a specific problem
record = get_record_by_problem_id(ds, '2063/B')
if record:
    print("Found problem:", record['title'])

# Quick lookup with nice display
quick_problem_lookup(ds, '2063/B')

# Explore all problems in contest 2063
explore_contest(ds, 2063)

# Search by title
results = search_problems_by_title(ds, 'binary', limit=5)
for problem_id, title in results:
    print(f"{problem_id}: {title}")

# List some available problem IDs
problem_ids = list_available_problem_ids(ds, limit=20)
print("Available problems:", problem_ids)
"""
"""
from datasets import load_dataset
test_ds = load_dataset("open-r1/codeforces", split="test")
print(f"Test problems: {len(test_ds)}")
print(type(test_ds[0]['contest_id']), test_ds[0]['contest_id'])

record = get_record_by_problem_id(test_ds, "2063/C")
if record:
    print("Found problem:", record["title"])
else:
    print("Not found")
list_available_problem_ids(test_ds, 500)
quick_problem_lookup(test_ds, '2063/C')
"""

def pretty_print_codeforces_problem_dark(problem_data: dict):
    """Pretty print a Codeforces problem with all fields and dark-mode styling."""
    if not isinstance(problem_data, dict):
        print("❌ Error: The provided problem data is not a valid dictionary.")
        return

    # --- MathJax ---
    mathjax = '''
    <script type="text/javascript" id="MathJax-script" async
      src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
    </script>
    <script>
    window.MathJax = {
      tex: { inlineMath: [['$', '$'], ['\\\\(', '\\\\)']], displayMath: [['$$', '$$']], processEscapes: true },
      options: { ignoreHtmlClass: ".*|", processHtmlClass: "arithmatex" }
    };
    </script>
    '''

    # --- Dark Mode Styles ---
    styles = '''
    <style>
        .codeforces-problem { font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; max-width: 900px; margin: 0 auto; color: #e0e0e0; }
        .header { background: linear-gradient(135deg, #111827 0%, #1f2937 100%); color: #e0e0e0; padding: 20px; border-radius: 12px; }
        .problem-title { font-size: 28px; font-weight: 700; margin: 0; color: #fefefe; }
        .contest-info { font-size: 14px; opacity: 0.8; margin-bottom: 12px; }
        .metadata { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 12px; }
        .metadata-item { background: rgba(255,255,255,0.1); padding: 6px 10px; border-radius: 8px; font-size: 13px; }
        .tags { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
        .tag { background: rgba(255,255,255,0.15); padding: 2px 8px; border-radius: 12px; font-size: 12px; text-transform: uppercase; }
        .section { padding: 18px 16px; border-bottom: 1px solid #374151; }
        .section-title { font-size: 18px; font-weight: 700; margin-bottom: 12px; color: #fefefe; }
        .section-content { line-height: 1.6; color: #d1d5db; }
        pre { background: #1e293b; padding: 10px; border-radius: 6px; overflow-x: auto; white-space: pre-wrap; word-break: break-word; color: #e0e0e0; }
        .example { margin-bottom: 12px; border: 1px solid #4b5563; border-radius: 6px; overflow: hidden; }
        .example-io { display: grid; grid-template-columns: 1fr 1fr; }
        .example-input h4, .example-output h4 { margin:0; padding:6px 12px; font-size:13px; font-weight:600; }
        .example-input h4 { background: #2563eb; color:#e0e0e0; }
        .example-output h4 { background: #16a34a; color:#e0e0e0; }
        @media (max-width: 700px) { .example-io { grid-template-columns: 1fr; } }
    </style>
    '''

    # --- Helper functions ---
    def render_value(val):
        if isinstance(val, dict):
            items = ''.join(f"<li><b>{html.escape(str(k))}:</b> {render_value(v)}</li>" for k,v in val.items())
            return f"<ul>{items}</ul>"
        elif isinstance(val, list):
            if all(isinstance(x, dict) and 'input' in x and 'output' in x for x in val):
                content = ''
                for ex in val:
                    content += f'''
                    <div class="example">
                        <div class="example-io">
                            <div class="example-input"><h4>Input</h4><pre>{html.escape(ex.get('input',''))}</pre></div>
                            <div class="example-output"><h4>Output</h4><pre>{html.escape(ex.get('output',''))}</pre></div>
                        </div>
                    </div>'''
                return content
            items = ''.join(f"<li>{render_value(x)}</li>" for x in val)
            return f"<ol>{items}</ol>"
        else:
            return html.escape(str(val))

    def convert_latex(text):
        import re
        if not text: return ''
        return re.sub(r'\$\$\$(.*?)\$\$\$', r'$$\1$$', text, flags=re.DOTALL)

    def render_section(title, content):
        if content is None: return ''
        converted = convert_latex(content) if isinstance(content, str) else content
        return f'''
        <div class="section">
            <div class="section-title">{html.escape(title)}</div>
            <div class="section-content">{render_value(converted)}</div>
        </div>'''

    # --- Header ---
    problem_id = problem_data.get('id','Unknown')
    title = problem_data.get('title','Unknown')
    contest_name = problem_data.get('contest_name','Unknown')
    index = problem_data.get('index','')
    time_limit = problem_data.get('time_limit',0)
    memory_limit = problem_data.get('memory_limit',0)
    rating = problem_data.get('rating',0)
    tags = problem_data.get('tags',[])
    contest_start = problem_data.get('contest_start', None)
    contest_date = ''
    if contest_start:
        try: contest_date = datetime.fromtimestamp(contest_start).strftime('%B %d, %Y at %H:%M UTC')
        except: contest_date = str(contest_start)

    tags_html = ''.join(f'<span class="tag">{html.escape(tag)}</span>' for tag in tags)
    header_html = f'''
    <div class="header">
        <div class="problem-title">{html.escape(title)}</div>
        <div class="contest-info">{html.escape(contest_name)} • Problem {html.escape(index)}</div>
        <div class="metadata">
            <div class="metadata-item">Time: {time_limit}s</div>
            <div class="metadata-item">Memory: {memory_limit}MB</div>
            <div class="metadata-item">Rating: {rating}</div>
            {f'<div class="metadata-item">Date: {html.escape(contest_date)}</div>' if contest_date else ''}
        </div>
        <div class="tags">{tags_html}</div>
    </div>
    '''

    # --- Sections ---
    sections_html = ''
    for key in problem_data.keys():
        if key in ['title','contest_name','index','time_limit','memory_limit','rating','tags','contest_start']: 
            continue
        sections_html += render_section(key, problem_data[key])

    # --- Full HTML ---
    full_html = f'''
    {mathjax}
    {styles}
    <div class="codeforces-problem">
        {header_html}
        {sections_html}
    </div>
    <script>
    function renderMathJax() {{
        if(window.MathJax && window.MathJax.typesetPromise) window.MathJax.typesetPromise().catch(e=>console.log(e));
    }}
    renderMathJax(); setTimeout(renderMathJax,100); setTimeout(renderMathJax,500); setTimeout(renderMathJax,1000);
    </script>
    '''

    display(HTML(full_html))


def pretty_print_codeforces_problem(problem_data: dict):
    """Pretty print a Codeforces problem including all fields, fully displaying nested tests and checker data."""
    
    if not isinstance(problem_data, dict):
        print("❌ Error: Provided data is not a dictionary.")
        return
    
    def fmt_date(ts):
        try:
            return datetime.fromtimestamp(ts).strftime('%B %d, %Y %H:%M UTC')
        except:
            return str(ts)
    
    def render_value(val):
        """Render arbitrary Python objects recursively into HTML."""
        if isinstance(val, dict):
            items = ''.join(f"<li><b>{html.escape(str(k))}:</b> {render_value(v)}</li>" for k, v in val.items())
            return f"<ul>{items}</ul>"
        elif isinstance(val, list):
            items = ''.join(f"<li>{render_value(v)}</li>" for v in val)
            return f"<ol>{items}</ol>"
        else:
            return html.escape(str(val))

        # Convert LaTeX from $$$ to $$
    def convert_latex(text):
        if not text:
            return ''
        import re
        return re.sub(r'\$\$\$(.*?)\$\$\$', r'$$\1$$', text, flags=re.DOTALL)
    # Build header
    html_parts = [f"<h2>{html.escape(str(problem_data.get('title', 'Unknown')))}</h2>"]
    
    # Metadata
    metadata_fields = ['id', 'contest_name', 'index', 'time_limit', 'memory_limit', 'rating', 'tags', 'contest_start', 'contest_type', 'executable', 'input_mode']
    metadata_html = ''
    for field in metadata_fields:
        val = problem_data.get(field)
        if field == 'contest_start' and val:
            val = fmt_date(val)
        if val is not None:
            metadata_html += f"<b>{html.escape(field)}:</b> {render_value(val)}<br>"
    html_parts.append(metadata_html)
    
    # Sections
    sections = ['description', 'input_format', 'output_format', 'interaction_format', 'note', 'editorial', 'solution']
    for sec in sections:
        val = problem_data.get(sec)
        if val:
            html_parts.append(f"<h3>{html.escape(sec)}</h3><div>{convert_latex(render_value(val))}</div>")

    # Examples
    examples = problem_data.get('examples')
    if examples:
        examples_html = ''
        for i, ex in enumerate(examples):
            in_text = ex.get('input','')
            out_text = ex.get('output','')
            examples_html += f"<div style='border:1px solid #ccc;padding:8px;margin:8px;'><b>Example {i+1}</b><br><b>Input:</b><pre>{html.escape(in_text)}</pre><b>Output:</b><pre>{html.escape(out_text)}</pre></div>"
        html_parts.append(f"<h3>Examples</h3>{examples_html}")
    
    # Other fields (including official_tests, generated_checker, generated_tests)
    other_fields = ['testset_size', 'official_tests', 'official_tests_complete', 'generated_checker', 'generated_tests']
    for field in other_fields:
        val = problem_data.get(field)
        if val is not None:
            html_parts.append(f"<h3>{html.escape(field)}</h3><div>{render_value(val)}</div>")
    
    display(HTML(''.join(html_parts)))


# Example usage:
def demo_codeforces_display():
    """Demo function showing how to use the Codeforces problem displayer"""
    sample_problem = {
        'id': '2046/A',
        'title': 'Swap Columns and Find a Path',
        'contest_name': 'Codeforces Round 990 (Div. 1)',
        'index': 'A',
        'time_limit': 2.0,
        'memory_limit': 512.0,
        'rating': 1200,
        'tags': ['greedy', 'sortings'],
        'contest_start': 1733207100,
        'description': "There is a matrix consisting of $$$2$$$ rows and $$$n$$$ columns...",
        'input_format': "Each test contains multiple test cases...",
        'output_format': "For each test case, print one integer...",
        'examples': [{'input': '3\n1\n-10\n5', 'output': '-5'}],
        'editorial': "We can divide the columns in the matrix into three different groups..."
    }
    
    pretty_print_codeforces_problem(sample_problem)
"""
if __name__ == "__main__":
    # Example of how to use it with your data
    demo_codeforces_display()
"""