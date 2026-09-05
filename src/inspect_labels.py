"""Peeks at raw JSONL label formats across domains before merging."""

import json

FILES = [
    "data/raw/difraud_phishing_train.jsonl",
    "data/raw/difraud_sms_train.jsonl",
]

for path in FILES:
    with open(path, "r") as f:
        first_line = json.loads(f.readline())
    print(path)
    print(first_line)
    print()