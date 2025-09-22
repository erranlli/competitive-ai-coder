import os
import argparse
import textwrap
from datasets import load_from_disk

# --- Configuration ---
DATASET_A_PATH = "/mnt/data2/filtered_datasets_flexible_match/successful_solutions"
DATASET_B_PATH = "/mnt/data2/new_deepcoder_cots_arrow_appexp" #new_deepcoder_cots_arrow_codeonly"

# python compare_single_problem.py --id_a "930/B" --id_b "deepcoder_6"

def find_and_print_problem(ds, problem_id, dataset_name):
    """Finds a problem by ID in a dataset and prints its contents."""
    print(f"\n{'='*30} Problem '{problem_id}' from {dataset_name} {'='*30}")
    
    # Efficiently find the row
    try:
        record = ds.filter(lambda example: example['id'] == problem_id, num_proc=4)[0]
    except IndexError:
        print(f"  ERROR: Problem ID '{problem_id}' not found in dataset.")
        return
    except Exception as e:
        print(f"  An error occurred while searching: {e}")
        return

    messages = record.get('messages', [{}, {}])
    
    # Basic structural validation
    if not (isinstance(messages, list) and len(messages) >= 2 and isinstance(messages[0], dict) and isinstance(messages[1], dict)):
        print("  ERROR: Message format is invalid for this record.")
        return

    user_prompt = messages[0].get('content', 'N/A')
    assistant_response = messages[1].get('content', 'N/A')
    
    wrapper = textwrap.TextWrapper(width=120, initial_indent="    ", subsequent_indent="    ")

    print(f"\n--- User Prompt (message[0], length:{len(user_prompt)}) ---")
    print("\n".join(wrapper.wrap(user_prompt)))

    print(f"\n--- Assistant Response (message[1], length:{len(assistant_response)}) ---")
    print("\n".join(wrapper.wrap(assistant_response)))
    print(f"\n{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Compare a specific problem from two different datasets.")
    parser.add_argument("--id_a", type=str, required=True, help="The ID of the problem to retrieve from Dataset A.")
    parser.add_argument("--id_b", type=str, required=True, help="The ID of the problem to retrieve from Dataset B.")
    args = parser.parse_args()

    print("Loading datasets...")
    try:
        ds_a = load_from_disk(DATASET_A_PATH, keep_in_memory=False)
        ds_b = load_from_disk(DATASET_B_PATH, keep_in_memory=False)
        print("Datasets loaded successfully.")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    find_and_print_problem(ds_a, args.id_a, f"Dataset A ({os.path.basename(DATASET_A_PATH)})")
    find_and_print_problem(ds_b, args.id_b, f"Dataset B ({os.path.basename(DATASET_B_PATH)})")


if __name__ == "__main__":
    main()
