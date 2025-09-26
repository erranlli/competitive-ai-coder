from datasets import load_dataset

dataset_id = "erranli/codeforces-cot-highquality"
local_path = "./codeforces-cot-highquality-dataset" # Directory where the dataset will be saved

# Load the dataset. The 'save_infos=True' argument can be useful for caching.
# You can also specify a split if you only want to download a particular part,
# e.g., split="train"
dataset = load_dataset(dataset_id)

# If you want to save it to disk in a specific format (e.g., as Parquet or JSON),
# you can iterate through the splits and save them.
# For example, to save to JSON files:
for split_name, split_dataset in dataset.items():
    split_dataset.to_json(f"{local_path}/{split_name}.json")
    print(f"Saved split '{split_name}' to {local_path}/{split_name}.json")

print(f"Dataset '{dataset_id}' downloaded and processed. Data might be cached locally by the datasets library, and optionally saved to {local_path}.")
