"""This script uploads the exported final model, its thresholds and the knowledge base to a Hugging Face model repository."""
import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
GITHUB = "https://github.com/Adossi-design/African-Medical-Question-Answering-Assistant"

# The model card replaces the card of the base model that Sentence-Transformers copies into the exported folder.
MODEL_CARD = f"""---
language: en
library_name: sentence-transformers
license: cc-by-nc-sa-4.0
base_model: sentence-transformers/multi-qa-MiniLM-L6-cos-v1
pipeline_tag: sentence-similarity
---

# African Medical Question Answering Assistant

This repository holds the files that the African Medical Question Answering Assistant loads when it starts. The model
is multi-qa-MiniLM-L6-cos-v1 fine-tuned with the Multiple Negatives Ranking loss on 711 question and answer pairs from
the Short Answer Questions of AfriMed-QA v2.5. The file `assistant_config.json` holds the two confidence thresholds, and
`knowledge_base.csv` holds the 1,205 unique human-written answers that the assistant can return.

The model is intended for educational use only and does not provide medical advice, diagnosis or treatment. The source
code, notebooks and evaluation are available at {GITHUB}.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="<username>/<model-name>")
    args = parser.parse_args()
    folder = ROOT / "models" / "final"
    assert (folder / "assistant_config.json").exists(), "Run notebook 04 first, since it exports models/final/."

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=args.repo, ignore_patterns=["README.md"],
                      commit_message="The fine-tuned model, its thresholds and the knowledge base are uploaded.")
    api.upload_file(path_or_fileobj=MODEL_CARD.encode("utf-8"), path_in_repo="README.md", repo_id=args.repo,
                    commit_message="The model card describes the fine-tuned model and its intended use.")
    print(f"Uploaded to https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
