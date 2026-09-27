"""Deploy Levi to a Hugging Face Space (Docker).

Uploads the current code through the Hub API instead of git, so binary sample
files need no Git LFS setup and the local history stays untouched. Secrets are
NOT uploaded: add them in the Space's Settings -> Variables and secrets.

Usage (after `hf auth login` with a write token):
    python scripts/deploy_space.py --repo <your-hf-username>/levi
"""
import argparse

from huggingface_hub import HfApi

# Everything the container needs; nothing local, secret or bulky.
IGNORE = [".env", ".venv/*", ".git/*", "data/*", "logs/*", "models_cache/*", "**/__pycache__/*", "*.pyc",
          ".pytest_cache/*", "eval/data/*", "eval/results/*.log", "eval/results/invalid/*", "logs_*.txt"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="<username>/<space-name>")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    api.create_repo(args.repo, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=".", repo_id=args.repo, repo_type="space", ignore_patterns=IGNORE,
                      commit_message="Deploy Levi")
    print(f"Uploaded. Build logs and app: https://huggingface.co/spaces/{args.repo}")
    print("Next: add GROQ_API_KEY (and optionally OPENROUTER_API_KEY, LANGFUSE_*) under Settings -> "
          "Variables and secrets, then restart the Space.")


if __name__ == "__main__":
    main()
