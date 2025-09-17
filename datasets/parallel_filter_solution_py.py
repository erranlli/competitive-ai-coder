import os
import re
import subprocess
import pathlib
from datasets import Dataset, load_dataset


# --- Configuration ---
# Path to your dataset
#DATASET_PATH = "open-r1/Mixture-of-Thoughts" 
#DATASET_SUBSET = "code"
DATASET_PATH = "open-r1/codeforces-cots" 
DATASET_SUBSET = "solutions_py"

# Where to save the final filtered datasets
OUTPUT_DIR = pathlib.Path("./filtered_datasets_flexible_match")
PROCESSED_DATA_DIR = OUTPUT_DIR / "processed_with_results"
SUCCESSFUL_DIR = OUTPUT_DIR / "successful_solutions"
FAILED_DIR = OUTPUT_DIR / "failed_solutions"

# Number of CPU cores to use
NUM_PROCESSES = os.cpu_count() - 2 if os.cpu_count() > 2 else 1

# Safety margin for timeout
TIMEOUT_SAFETY_MARGIN = 1.0 

# --- NEW: Flexible Matching Configuration ---
# Set to True to sort numbers on each line before comparing.
# This handles cases where the order of output numbers doesn't matter.
ORDER_INVARIANT_CHECK = True
# Precision for comparing floating-point numbers. 
# 1.0 vs 1.000 will be considered equal.
FLOAT_PRECISION = 6

# --- Helper Functions ---

def get_canonical_form_of_line(line: str):
    """
    Converts a line of output into a canonical representation for flexible comparison.
    - Handles different float precisions (e.g., 1.0 vs 1.000)
    - Optionally handles order-invariance of numbers on a line (e.g., "1 2 3" vs "3 2 1")
    """
    tokens = line.strip().split()
    processed_tokens = []
    
    try:
        # Attempt to convert all tokens to floats for numerical comparison
        for token in tokens:
            processed_tokens.append(round(float(token), FLOAT_PRECISION))
    except ValueError:
        # If any token is not a number (e.g., "YES"), treat the whole line as strings
        processed_tokens = tokens

    # If order doesn't matter, sort the tokens to create a canonical order
    if ORDER_INVARIANT_CHECK:
        processed_tokens.sort()
        
    return processed_tokens

def are_outputs_equivalent(actual_lines, expected_lines):
    """
    Compares two blocks of output flexibly based on the configuration.
    """
    # Must have the same number of non-empty lines
    if len(actual_lines) != len(expected_lines):
        return False
        
    # Compare each line in its canonical form
    for actual_line, expected_line in zip(actual_lines, expected_lines):
        if get_canonical_form_of_line(actual_line) != get_canonical_form_of_line(expected_line):
            return False
            
    return True

def extract_python_code(messages):
    if not messages: return None
    assistant_messages = [msg['content'] for msg in messages if msg.get('role') == 'assistant']
    if not assistant_messages: return None
    last_message = assistant_messages[-1]
    code_match = re.search(r"```python\n(.*?)```", last_message, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    return None

def normalize_output(text):
    if not isinstance(text, str): return []
    # This function now just cleans up lines, the main logic is elsewhere
    return [line.strip() for line in text.strip().splitlines() if line.strip()]

def run_code_with_test_cases(code, examples, time_limit):
    """
    Runs code, now with flexible output comparison.
    """
    if not code or not examples:
        return {"passed_all": False, "reason": "No code or test cases found"}

    execution_timeout = time_limit + TIMEOUT_SAFETY_MARGIN
    for i, test_case in enumerate(examples):
        test_input = test_case.get('input', '')
        expected_output_str = test_case.get('output', '')
        try:
            process = subprocess.run(
                ['python', '-c', code],
                input=test_input,
                capture_output=True,
                text=True,
                timeout=execution_timeout
            )
            if process.returncode != 0:
                return {"passed_all": False, "reason": f"Runtime Error on test {i+1}", "stderr": process.stderr}
            
            actual_lines = normalize_output(process.stdout)
            expected_lines = normalize_output(expected_output_str)
            
            # --- REVISED COMPARISON LOGIC ---
            if not are_outputs_equivalent(actual_lines, expected_lines):
                return {"passed_all": False, "reason": f"Wrong Answer on test {i+1}", "input": test_input, "expected": expected_lines, "got": actual_lines}

        except subprocess.TimeoutExpired:
            return {"passed_all": False, "reason": f"Time Limit Exceeded on test {i+1} (Limit: {time_limit}s, Ran for >{execution_timeout}s)"}
        except Exception as e:
            return {"passed_all": False, "reason": f"Execution failed on test {i+1} with error: {e}"}
            
    return {"passed_all": True, "reason": "All tests passed"}

def process_and_verify(example):
    """Mapped function to verify a single example."""
    solution_code = extract_python_code(example['messages'])
    test_cases = example['examples']
    problem_time_limit = example.get('time_limit')

    if not solution_code:
        result = {"passed_all": False, "reason": "Could not extract Python code."}
    elif not test_cases:
        result = {"passed_all": False, "reason": "No test cases found in 'examples' field."}
    elif problem_time_limit is None:
        result = {"passed_all": False, "reason": "time_limit field is missing."}
    else:
        result = run_code_with_test_cases(solution_code, test_cases, problem_time_limit)
    
    return {"verification_result": result}

# --- Main Execution Logic (remains the same) ---

if __name__ == "__main__":
    print("Starting the PARALLEL dataset processing script with FLEXIBLE MATCHING.")
    print(f"Order-invariant check: {ORDER_INVARIANT_CHECK}, Float precision: {FLOAT_PRECISION}")
    print(f"Using {NUM_PROCESSES} worker processes.")

    # 1. Load the dataset
    print(f"Loading dataset '{DATASET_PATH}' with subset '{DATASET_SUBSET}'...")
    original_dataset = load_dataset(DATASET_PATH, DATASET_SUBSET, split="train")
    print(f"Dataset loaded successfully. Number of rows: {len(original_dataset)}")

    # Create directories
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # 2. Map the processing function in parallel
    print("\nExecuting solutions against test cases in parallel...")
    processed_dataset = original_dataset.map(
        process_and_verify,
        num_proc=NUM_PROCESSES
    )
    
    print("\nParallel processing complete.")
    
    # Optional: Save intermediate dataset
    print(f"Saving processed dataset with verification results to '{PROCESSED_DATA_DIR}'...")
    processed_dataset.save_to_disk(str(PROCESSED_DATA_DIR))

    # 3. Filter and save final datasets
    print("Filtering into successful and failed datasets...")
    successful_dataset = processed_dataset.filter(lambda x: x['verification_result']['passed_all'])
    failed_dataset = processed_dataset.filter(lambda x: not x['verification_result']['passed_all'])
    
    print(f"Saving successful solutions to '{SUCCESSFUL_DIR}'...")
    successful_dataset.save_to_disk(str(SUCCESSFUL_DIR))
    
    print(f"Saving failed solutions to '{FAILED_DIR}'...")
    failed_dataset.save_to_disk(str(FAILED_DIR))

    print("\n--- SCRIPT FINISHED ---")
    print(f"  ✅ Successful solutions: {len(successful_dataset)}")
    print(f"  ❌ Failed solutions:     {len(failed_dataset)}")
    print(f"\nTo load your filtered data, for example:")
    print(f"successful_ds = load_from_disk('{SUCCESSFUL_DIR}')")