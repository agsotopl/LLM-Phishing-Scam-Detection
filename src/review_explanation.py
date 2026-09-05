"""Prints a few sample explanations for eyeballing"""

import json

with open("outputs/explanations.jsonl", "r") as f:
    records = [json.loads(line) for line in f]

disagreements = [r for r in records if r["true_label"] != r["predicted_label"]]
agreements = [r for r in records if r["true_label"] == r["predicted_label"]]

print(f"{len(disagreements)} disagreements out of {len(records)} total\n")

print("--- sample disagreements ---")
for r in disagreements[:3]:
    print(r)
    print()

print("--- sample agreements ---")
for r in agreements[:2]:
    print(r)
    print()