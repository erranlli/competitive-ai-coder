import os
import difflib
from datasets import load_from_disk
import textwrap
import numpy as np
from transformers import AutoTokenizer
import mistune
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import html
from pygments.util import ClassNotFound
import re

# --- Configuration ---
DATASET_A_PATH = "/mnt/data2/filtered_datasets_flexible_match/successful_solutions"
#DATASET_B_PATH = "/mnt/data2/new_deepcoder_cots_arrow_codeonly"
DATASET_B_PATH = "/mnt/data2/new_deepcoder_cots_arrow_appexp"
NUM_SAMPLES_TO_PRINT = 3
TOKENIZER_NAME = "Qwen/Qwen2.5-7B-Instruct"

def print_percentiles(data, title):
    """Helper function to print percentile stats for a list of numbers."""
    print(f"\n--- {title} ---")
    if not data:
        print("  No data to display.")
        return
    
    p = np.percentile(data, [25, 50, 75, 95])
    print(f"  - 25th Percentile: {p[0]:.0f}")
    print(f"  - 50th Percentile (Median): {p[1]:.0f}")
    print(f"  - 75th Percentile: {p[2]:.0f}")
    print(f"  - 95th Percentile: {p[3]:.0f}")
    print(f"  - Maximum: {np.max(data):.0f}")

def analyze_full_dataset_lengths(ds, tokenizer):
    """Analyzes and returns length statistics for an entire dataset."""
    lengths = {'msg0_chars': [], 'msg0_tokens': [], 'msg1_chars': [], 'msg1_tokens': []}
    for row in ds:
        messages = row.get('messages', [{}, {}])
        if not (isinstance(messages, list) and len(messages) >= 2 and isinstance(messages[0], dict) and isinstance(messages[1], dict)):
            continue
        msg0, msg1 = messages[0].get('content', ''), messages[1].get('content', '')
        lengths['msg0_chars'].append(len(msg0))
        lengths['msg1_chars'].append(len(msg1))
        if tokenizer:
            lengths['msg0_tokens'].append(len(tokenizer.encode(msg0, add_special_tokens=False)))
            lengths['msg1_tokens'].append(len(tokenizer.encode(msg1, add_special_tokens=False)))
    return lengths

def analyze_approach_explanation(ds):
    """Analyzes how many samples contain 'Approach' and 'Explanation' sections after </think>."""
    counts = {'approach': 0, 'explanation': 0, 'both': 0}
    think_splitter = re.compile(r"</think>", flags=re.IGNORECASE)
    approach_re = re.compile(r"(^|\n)\s*#{0,6}\s*approach\b", flags=re.IGNORECASE)
    explanation_re = re.compile(r"(^|\n)\s*#{0,6}\s*explanation\b", flags=re.IGNORECASE)

    for row in ds:
        messages = row.get('messages', [{}, {}])
        if not (isinstance(messages, list) and len(messages) >= 2 and isinstance(messages[1], dict)):
            continue
        
        assistant_response = messages[1].get('content', '')
        parts = think_splitter.split(assistant_response, maxsplit=1)
        if len(parts) < 2:
            continue
        after_think = parts[1]

        has_approach = approach_re.search(after_think) is not None
        has_explanation = explanation_re.search(after_think) is not None

        if has_approach:
            counts['approach'] += 1
        if has_explanation:
            counts['explanation'] += 1
        if has_approach and has_explanation:
            counts['both'] += 1
    return counts

# --- Main Script ---
def main():
    """
    Computes statistics on two datasets, handling both common and disjoint problem sets.
    """
    print("Loading datasets...")
    try:
        ds_a = load_from_disk(DATASET_A_PATH, keep_in_memory=False)
        ds_b = load_from_disk(DATASET_B_PATH, keep_in_memory=False)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        tokenizer = None

    print("\n--- Basic Stats ---")
    print(f"Dataset A ('{os.path.basename(DATASET_A_PATH)}'): {len(ds_a)} samples")
    print(f"Dataset B ('{os.path.basename(DATASET_B_PATH)}'): {len(ds_b)} samples")

    # --- Approach/Explanation Analysis ---
    print("\nAnalyzing for 'Approach' and 'Explanation' sections...")
    counts_a = analyze_approach_explanation(ds_a)
    counts_b = analyze_approach_explanation(ds_b)
    
    print(f"\n--- Dataset A ('{os.path.basename(DATASET_A_PATH)}') ---")
    print(f"  - Samples with 'Approach' after </think>: {counts_a['approach']} ({counts_a['approach']/len(ds_a):.2%})")
    print(f"  - Samples with 'Explanation' after </think>: {counts_a['explanation']} ({counts_a['explanation']/len(ds_a):.2%})")
    print(f"  - Samples with BOTH sections after </think>: {counts_a['both']} ({counts_a['both']/len(ds_a):.2%})")
    
    print(f"\n--- Dataset B ('{os.path.basename(DATASET_B_PATH)}') ---")
    print(f"  - Samples with 'Approach' after </think>: {counts_b['approach']} ({counts_b['approach']/len(ds_b):.2%})")
    print(f"  - Samples with 'Explanation' after </think>: {counts_b['explanation']} ({counts_b['explanation']/len(ds_b):.2%})")
    print(f"  - Samples with BOTH sections after </think>: {counts_b['both']} ({counts_b['both']/len(ds_b):.2%})")


    # --- Full Dataset Length Analysis ---
    print("\n\nAnalyzing length distributions for all samples in each dataset...")
    
    lengths_a = analyze_full_dataset_lengths(ds_a, tokenizer)
    lengths_b = analyze_full_dataset_lengths(ds_b, tokenizer)

    print_percentiles(lengths_a['msg0_chars'], "Dataset A - User Prompt Length (chars)")
    if tokenizer: print_percentiles(lengths_a['msg0_tokens'], "Dataset A - User Prompt Length (tokens)")
    print_percentiles(lengths_a['msg1_chars'], "Dataset A - Assistant Response Length (chars)")
    if tokenizer: print_percentiles(lengths_a['msg1_tokens'], "Dataset A - Assistant Response Length (tokens)")

    print_percentiles(lengths_b['msg0_chars'], "Dataset B - User Prompt Length (chars)")
    if tokenizer: print_percentiles(lengths_b['msg0_tokens'], "Dataset B - User Prompt Length (tokens)")
    print_percentiles(lengths_b['msg1_chars'], "Dataset B - Assistant Response Length (chars)")
    if tokenizer: print_percentiles(lengths_b['msg1_tokens'], "Dataset B - Assistant Response Length (tokens)")

    # --- Common Problem Analysis ---
    print("\n\n--- Common Problem Analysis ---")
    ids_a = set(ds_a['id'])
    ids_b = set(ds_b['id'])
    common_ids = sorted(list(ids_a.intersection(ids_b)))
    num_common = len(common_ids)
    print(f"Number of common problems found: {num_common}")

    if not common_ids:
        print("\nNo common problems found. Printing first few samples from each dataset for general comparison.")
        
        wrapper = textwrap.TextWrapper(width=100, initial_indent="    ", subsequent_indent="    ")
        
        print(f"\n--- First {NUM_SAMPLES_TO_PRINT} Samples from Dataset A ---")
        for i, row in enumerate(ds_a):
            if i >= NUM_SAMPLES_TO_PRINT: break
            print(f"\n{'='*20} Sample {i+1} | ID: {row.get('id', 'N/A')} {'='*20}")
            messages = row.get('messages', [])
            if messages:
                print("  User Message (message[0]):")
                print("\n".join(wrapper.wrap(messages[0].get('content', 'N/A'))))
                print("\n  Assistant Message (message[1]):")
                print("\n".join(wrapper.wrap(messages[1].get('content', 'N/A'))))

        print(f"\n\n--- First {NUM_SAMPLES_TO_PRINT} Samples from Dataset B ---")
        for i, row in enumerate(ds_b):
            if i >= NUM_SAMPLES_TO_PRINT: break
            print(f"\n{'='*20} Sample {i+1} | ID: {row.get('id', 'N/A')} {'='*20}")
            messages = row.get('messages', [])
            if messages:
                print("  User Message (message[0]):")
                print("\n".join(wrapper.wrap(messages[0].get('content', 'N/A'))))
                print("\n  Assistant Message (message[1]):")
                print("\n".join(wrapper.wrap(messages[1].get('content', 'N/A'))))

        return

    # --- If common problems exist, proceed with diffing ---
    print("\nAnalyzing differences in common problems...")
    map_a = {row['id']: row for row in ds_a if row['id'] in common_ids}
    map_b = {row['id']: row for row in ds_b if row['id'] in common_ids}
    
    identical_msg0_count = 0
    identical_msg1_count = 0
    identical_both_count = 0
    different_problem_ids = []

    for problem_id in common_ids:
        row_a = map_a.get(problem_id)
        row_b = map_b.get(problem_id)
        
        if not row_a or not row_b: continue
        
        messages_a = row_a.get('messages', [{}, {}])
        messages_b = row_b.get('messages', [{}, {}])

        if not (isinstance(messages_a, list) and len(messages_a) >= 2 and isinstance(messages_a[0], dict) and isinstance(messages_a[1], dict)): continue
        if not (isinstance(messages_b, list) and len(messages_b) >= 2 and isinstance(messages_b[0], dict) and isinstance(messages_b[1], dict)): continue

        prompt_a = messages_a[0].get('content', '')
        prompt_b = messages_b[0].get('content', '')
        response_a = messages_a[1].get('content', '')
        response_b = messages_b[1].get('content', '')

        msg0_is_same = (prompt_a == prompt_b)
        msg1_is_same = (response_a == response_b)

        if msg0_is_same: identical_msg0_count += 1
        if msg1_is_same: identical_msg1_count += 1
        if msg0_is_same and msg1_is_same: identical_both_count += 1
        else: different_problem_ids.append(problem_id)

    print("\n--- Statistics for Common Problems ---")
    print(f"Total common problems: {num_common}")
    print(f"Problems with identical user prompts (message[0]): {identical_msg0_count} ({identical_msg0_count/num_common:.2%})")
    print(f"Problems with identical assistant responses (message[1]): {identical_msg1_count} ({identical_msg1_count/num_common:.2%})")
    print(f"Problems with both messages identical: {identical_both_count} ({identical_both_count/num_common:.2%})")

    if not different_problem_ids:
        print("\nNo differing samples found to display.")
    else:
        print(f"\n\n--- Showing Unified Diff for First {NUM_SAMPLES_TO_PRINT} Different Samples ---")
        print("---")
        print("Legend: (-) from Dataset A (working_old), (+) from Dataset B (failing_new)")
        print("---")

        for i, problem_id in enumerate(different_problem_ids[:NUM_SAMPLES_TO_PRINT]):
            print(f"\n\n{'='*30} Sample {i+1} | Problem ID: {problem_id} {'='*30}")
            
            row_a = map_a.get(problem_id)
            row_b = map_b.get(problem_id)
            
            prompt_a = row_a['messages'][0].get('content', 'N/A')
            prompt_b = row_b['messages'][0].get('content', 'N/A')
            response_a = row_a['messages'][1].get('content', 'N/A')
            response_b = row_b['messages'][1].get('content', 'N/A')
            
            print("\n--- Diff for User Prompt (message[0]) ---")
            prompt_diff = difflib.unified_diff(
                prompt_a.splitlines(keepends=True), prompt_b.splitlines(keepends=True),
                fromfile="Dataset A", tofile="Dataset B", lineterm='\\n'
            )
            diff_str = "".join(prompt_diff)
            if not diff_str.strip():
                print("(User prompts are identical for this sample)")
            else:
                print(diff_str)

            print("\n--- Diff for Assistant Response (message[1]) ---")
            response_diff = difflib.unified_diff(
                response_a.splitlines(keepends=True), response_b.splitlines(keepends=True),
                fromfile="Dataset A", tofile="Dataset B", lineterm='\\n'
            )
            diff_str = "".join(response_diff)
            if not diff_str.strip():
                print("(Assistant responses are identical for this sample)")
            else:
                print(diff_str)

if __name__ == "__main__":
    main()
