#!/usr/bin/env python3
"""
Upload a local Hugging Face model directory and its checkpoints to the Hub.

The script automatically determines the user's namespace (username) from the
Hugging Face token and creates the repository under that user.

The main model directory is uploaded to the 'main' branch. If the
--upload-all-checkpoints flag is used, each 'checkpoint-*' subdirectory
is uploaded to its own branch.

Example:
  python upload_hf/upload_model_to_hub.py \
    --run-dir /mnt/data2/qwen2.5-7b-mot-full-run0 \
    --repo-name qwen2.5-7b-mot-full \
    --upload-all-checkpoints \
    --private

Auth:
  - Ensure your HF token is available via `huggingface-cli login` or env HF_TOKEN.
"""

import argparse
import os
import sys

from huggingface_hub import HfApi, create_repo, HfFolder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload a local model directory and checkpoints to the Hugging Face Hub")
    p.add_argument("--run-dir", required=True, help="Path to the local training run directory (contains model files and optional 'checkpoint-*' subdirs)")
    p.add_argument("--repo-name", required=True, help="Target repository name on the Hub (e.g., 'coder-ft-qwen2.5-7b')")
    p.add_argument("--private", action="store_true", help="Create/update a private repo (default: public)")
    p.add_argument("--token", default=None, help="HF token (optional; defaults to HF_TOKEN or cached token)")
    p.add_argument("--commit-message", default="Upload model files", help="Commit message for the upload")
    p.add_argument("--allow-patterns", nargs="*", default=None, help="Whitelist patterns to include (optional)")
    p.add_argument("--ignore-patterns", nargs="*", default=None, help="Patterns to exclude (optional)")
    p.add_argument("--upload-all-checkpoints", action="store_true", help="Upload all 'checkpoint-*' subdirectories found in --run-dir to separate branches.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        print(f"Error: --run-dir does not exist or is not a directory: {run_dir}")
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
        print(f"Error: Could not retrieve user info from token. Please ensure your token is valid. Details: {e}")
        sys.exit(4)

    repo_id = f"{username}/{args.repo_name}"

    try:
        print(f"Ensuring repository '{repo_id}' exists...")
        create_repo(repo_id=repo_id, token=token, private=bool(args.private), repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"Warning: create_repo failed (might already exist): {e}")

    uploads_to_process = []
    uploads_to_process.append((run_dir, "main"))

    if args.upload_all_checkpoints:
        print("Searching for checkpoints to upload...")
        for item in sorted(os.listdir(run_dir)):
            full_path = os.path.join(run_dir, item)
            if os.path.isdir(full_path) and item.startswith("checkpoint-"):
                uploads_to_process.append((full_path, item))
                print(f"  - Found checkpoint: {item}")

    for local_dir, branch_name in uploads_to_process:
        print("\n" + "="*50)
        print(f"Uploading '{local_dir}'")
        print(f"  to repository: '{repo_id}'")
        print(f"  to branch:     '{branch_name}'")
        print("="*50)

        # --- FIX: CREATE THE BRANCH IF IT'S NOT 'main' ---
        # The 'main' branch is created by `create_repo`, but other branches must be created explicitly.
        if branch_name != "main":
            try:
                print(f"Ensuring branch '{branch_name}' exists on the Hub...")
                api.create_branch(
                    repo_id=repo_id,
                    branch=branch_name,
                    repo_type="model",
                    exist_ok=True, # Don't raise an error if the branch already exists
                )
            except Exception as e:
                # We print a warning but continue, in case the error is temporary
                print(f"⚠️  Could not create branch '{branch_name}'. The upload may fail. Error: {e}")
        # --- END FIX ---

        try:
            api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=local_dir,
                allow_patterns=args.allow_patterns,
                ignore_patterns=args.ignore_patterns,
                commit_message=f"{args.commit_message} (branch: {branch_name})",
                revision=branch_name,
            )
            print(f"✅ Successfully uploaded branch '{branch_name}'.")
        except Exception as e:
            print(f"❌ Upload for branch '{branch_name}' failed: {e}")

    print("\nAll uploads attempted.")
    repo_url = f"https://huggingface.co/{repo_id}"
    print(f"View your repository here: {repo_url}")


if __name__ == "__main__":
    main()