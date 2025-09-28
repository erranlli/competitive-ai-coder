from huggingface_hub import snapshot_download

dataset_id = "erranli/codeforces-cot-highquality"
local_dir = "./codeforces-cot-highquality-raw-files" # The directory where you want to save the original files

snapshot_download(repo_id=dataset_id, local_dir=local_dir, repo_type="dataset")

print(f"All raw files from dataset '{dataset_id}' downloaded to {local_dir}")
