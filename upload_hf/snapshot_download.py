from huggingface_hub import snapshot_download

repo_id = "erranli/rl-qwen2.5-7b-mot"
local_dir = "./rl-qwen2.5-7b-mot-full-repo" # The directory where you want to save all files

snapshot_download(repo_id=repo_id, local_dir=local_dir)

print(f"All files from {repo_id} downloaded to {local_dir}")
