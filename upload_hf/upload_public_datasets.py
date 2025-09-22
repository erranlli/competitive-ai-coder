import os
from pathlib import Path
from huggingface_hub import HfApi, HfFolder

def upload_selective_datasets():
    """
    Uploads specific file patterns from local directories to Hugging Face Hub repositories.
    This avoids uploading temporary or cache files.
    """
    # --- Configuration ---
    dataset_mappings = [
        {
            "local_path": Path("/mnt/data2/new_deepcoder_cots_arrow_appexp"),
            "repo_id": "erranli/deepcoder-train-success",
            "glob_pattern": "data-*.arrow",  # <-- THIS LINE IS THE FIX
            "is_private": False 
        },
        {
            "local_path": Path("/mnt/data2/deepcoder_trajectories"),
            "repo_id": "erranli/deepcoder-train-traj",
            "is_private": False
        },
        {
            "local_path": Path("/mnt/data2/filtered_datasets_flexible_match/successful_solutions/"),
            "repo_id": "erranli/codeforces-cot-filtered",
            "is_private": False
        }
    ]

    # --- Script Logic ---
    print("Initializing Hugging Face API...")
    api = HfApi()

    token = HfFolder.get_token()
    if token is None:
        print("\nError: Hugging Face token not found. Please log in first.")
        return

    print("Login successful. Starting SELECTIVE upload process...\n") # Message changed for clarity

    for mapping in dataset_mappings:
        local_path = mapping["local_path"]
        repo_id = mapping["repo_id"]
        is_private = mapping["is_private"]
        glob_pattern = mapping.get("glob_pattern")
        
        print("-" * 50)
        print(f"Processing dataset: {repo_id}")
        
        if not local_path.exists() or not local_path.is_dir():
            print(f"  [SKIPPING] Local path not found: {local_path}")
            continue

        try:
            print(f"  Creating repository '{repo_id}' on the Hub...")
            api.create_repo(repo_id=repo_id, repo_type="dataset", private=is_private, exist_ok=True)
            print(f"  Repository handling complete.")
        except Exception as e:
            print(f"  [ERROR] Could not create repository '{repo_id}': {e}")
            continue

        try:
            # This 'if' block is the most important part
            if glob_pattern:
                print(f"  Uploading ONLY files matching '{glob_pattern}' from '{local_path}'...")
                api.upload_folder(
                    folder_path=str(local_path),
                    repo_id=repo_id,
                    repo_type="dataset",
                    allow_patterns=glob_pattern, # This tells it to filter
                    commit_message=f"Upload clean data files"
                )
            else:
                print(f"  Uploading all files from '{local_path}'...")
                api.upload_folder(
                    folder_path=str(local_path),
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message=f"Initial upload of dataset"
                )
            print(f"  [SUCCESS] Successfully uploaded to {repo_id}")
        except Exception as e:
            print(f"  [ERROR] Failed to upload files for '{repo_id}': {e}")
            
        print("-" * 50 + "\n")

    print("All datasets processed.")


if __name__ == "__main__":
    upload_selective_datasets()