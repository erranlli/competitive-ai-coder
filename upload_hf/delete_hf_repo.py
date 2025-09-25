import argparse
from huggingface_hub import HfApi, HfFolder
from getpass import getpass

def main():
    parser = argparse.ArgumentParser(description="Delete a Hugging Face repository.")
    parser.add_argument("repo_id", type=str, help="The ID of the repository to delete (e.g., 'username/repo-name').")
    args = parser.parse_args()

    token = HfFolder.get_token()
    if not token:
        token = getpass("Enter your Hugging Face write token: ")

    api = HfApi()

    print(f"You are about to permanently delete the repository: {args.repo_id}")
    confirm = input("Are you sure? This action is irreversible. [y/N] ")

    if confirm.lower() == 'y':
        try:
            api.delete_repo(repo_id=args.repo_id, token=token)
            print(f"Successfully deleted repository: {args.repo_id}")
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print("Deletion cancelled.")

if __name__ == "__main__":
    main()
