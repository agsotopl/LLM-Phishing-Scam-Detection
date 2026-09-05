"""Merges DIFrauD phishing and sms raw JSONL into one cleaned dataframe"""

import json
import os

import ftfy
import pandas as pd

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
CONFIGS = ["phishing", "sms"]
SPLITS = ["train", "test", "validation"]


def load_jsonl(path):
    """Reads a JSONL file into a list of dicts."""
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def build_dataframe(raw_dir=RAW_DIR):
    """Loads all domain/split JSONL files into one tagged, merged dataframe."""
    rows = []
    for config in CONFIGS:
        domain = "email" if config == "phishing" else "sms"
        for split in SPLITS:
            path = os.path.join(raw_dir, f"difraud_{config}_{split}.jsonl")
            records = load_jsonl(path)
            for r in records:
                rows.append({
                    "text": r["text"],
                    "label": r["label"],
                    "domain": domain,
                    "split": split,
                })
    return pd.DataFrame(rows)


def clean_text(df):
    """Fixes mojibake, strips whitespace, and drops empty or duplicate rows."""
    df["text"] = df["text"].apply(ftfy.fix_text).str.strip()
    df = df[df["text"].str.len() > 0]
    before = len(df)
    df = df.drop_duplicates(subset="text")
    print(f"dropped {before - len(df)} duplicate rows")
    return df.reset_index(drop=True)


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df = build_dataframe()
    df = clean_text(df)

    print(df["domain"].value_counts())
    print(df["split"].value_counts())
    print(df["label"].value_counts())

    out_path = os.path.join(PROCESSED_DIR, "difraud_merged.parquet")
    df.to_parquet(out_path, index=False)
    print(f"saved {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()