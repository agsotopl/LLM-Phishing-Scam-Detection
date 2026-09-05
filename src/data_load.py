"""Downloads DIFrauD phishing (email) and sms JSONL files directly from the Hugging Face repo"""

import os
from huggingface_hub import hf_hub_download

REPO_ID = "redasers/difraud"
RAW_DIR = "data/raw"
CONFIGS = ["phishing", "sms"]
SPLITS = ["train", "test", "validation"]


def download_difraud(raw_dir=RAW_DIR):
    """Downloads each DIFrauD config/split JSONL and copies it into raw_dir."""
    os.makedirs(raw_dir, exist_ok=True)

    for config in CONFIGS:
        for split in SPLITS:
            filename = f"{config}/{split}.jsonl"
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="dataset",
            )
            out_path = os.path.join(raw_dir, f"difraud_{config}_{split}.jsonl")
            with open(local_path, "rb") as src, open(out_path, "wb") as dst:
                dst.write(src.read())
            print(f"saved {out_path}")


if __name__ == "__main__":
    download_difraud()