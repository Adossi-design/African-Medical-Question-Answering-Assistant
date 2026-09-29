"""Publish the web app and the final model to a Hugging Face Space.

Before the first upload, the account owner installs `huggingface_hub` with pip and runs `huggingface-cli login`
with a token that has write permission (created at huggingface.co/settings/tokens). Then, from the project root and
after notebook 04 has exported `models/final/`, run:

    python deploy/upload_to_space.py --space <your-username>/african-medical-qa-assistant
"""
import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
APP_FILES = ["app.py", "src/__init__.py", "src/config.py", "src/text_cleaning.py", "src/retrieval.py", "src/assistant.py"]

# Hugging Face Spaces reads its settings from the README of the Space, so it is written here at upload time.
SPACE_README = """---
title: African Medical QA Assistant
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: app.py
pinned: false
license: cc-by-nc-sa-4.0
short_description: Educational medical question answering based on AfriMed-QA
---

This assistant answers medical questions with human-written answers from the Short Answer Questions of the
AfriMed-QA v2.5 dataset. It is for educational use only and does not provide medical advice, diagnosis or treatment.
The source code, notebooks and full documentation are in the project's GitHub repository.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", required=True, help="<username>/<space-name>")
    args = parser.parse_args()
    assert (ROOT / "models/final/assistant_config.json").exists(), "Run notebook 04 first, since it exports models/final/."

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for f in APP_FILES:
            (stage / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / f, stage / f)
        shutil.copytree(ROOT / "models/final", stage / "models/final")
        (stage / "README.md").write_text(SPACE_README, encoding="utf-8")
        shutil.copy(ROOT / "deploy/requirements.txt", stage / "requirements.txt")

        api = HfApi()
        api.create_repo(args.space, repo_type="space", space_sdk="gradio", exist_ok=True)
        api.upload_folder(folder_path=str(stage), repo_id=args.space, repo_type="space",
                          commit_message="Deploy African Medical QA Assistant")
    print(f"Uploaded to https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
