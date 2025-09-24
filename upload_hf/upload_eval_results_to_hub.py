#!/usr/bin/env python3
"""
Upload a local results directory to a Hugging Face Dataset repository.

Example:
  python upload_hf/upload_eval_results_to_hub.py \
    --results-dir ./results_best_saved/qwen-qwen2.5-7b-instruct__open-r1-codeforces__default__test__vllm/ \
    --dataset-name qwen2.5-7b-codeforces-benchmark-results \
    --private

Auth:
  - Ensure your HF token is available via `huggingface-cli login` or env HF_TOKEN.
"""

import argparse
import os
import sys

from huggingface_hub import HfApi, create_repo, HfFolder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload a local results directory to a Hugging Face Dataset repository")
    p.add_argument("--results-dir", required=True, help="Path to the local directory containing the results files (e.g., .jsonl)")
    p.add_argument("--dataset-name", required=True, help="Target repository name on the Hub for this dataset (e.g., 'my-benchmark-results')")
    p.add_argument("--private", action="store_true", help="Create a private dataset repo (default: public)")
    p.add_argument("--token", default=None, help="HF token (optional; defaults to HF_TOKEN or cached token)")
    p.add_argument("--commit-message", default="Upload evaluation results", help="Commit message for the upload")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    results_dir = os.path.abspath(args.results_dir)
    if not os.path.isdir(results_dir):
        print(f"Error: --results-dir does not exist or is not a directory: {results_dir}")
        sys.exit(1)

    token = args.token or os.environ.get("HF_TOKEN") or HfFolder.get_token()
    if not token:
        print("Error: No HF token found. Run `huggingface-cli login` or set HF_TOKEN env, or pass --token.")
        sys.exit(2)

    api = HfApi(token=token)

    try:
        user_info = api.whoami()
        username = user_info['name']
        print(f"Authenticated as user: '{username}'")
    except Exception as e:
        print(f"Error: Could not retrieve user info from token. Details: {e}")
        sys.exit(4)
        
    # Construct the full repo_id
    repo_id = f"{username}/{args.dataset_name}"

    try:
        print(f"Ensuring dataset repository '{repo_id}' exists...")
        # CRITICAL: Set repo_type to "dataset"
        create_repo(repo_id=repo_id, token=token, private=bool(args.private), repo_type="dataset", exist_ok=True)
    except Exception as e:
        print(f"Warning: create_repo failed (might already exist): {e}")

    print(f"\nUploading '{results_dir}' → '{repo_id}'")
    try:
        api.upload_folder(
            repo_id=repo_id,
            # CRITICAL: Specify repo_type
            repo_type="dataset",
            folder_path=results_dir,
            commit_message=args.commit_message,
        )
        print("✅ Upload complete.")
        dataset_url = f"https://huggingface.co/datasets/{repo_id}"
        print(f"View your dataset here: {dataset_url}")

    except Exception as e:
        print(f"❌ Upload failed: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()